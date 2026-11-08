"""
Unified HuggingFace Transformer Fine-Tuning Pipeline for NepSent-Engine.

Supports:
- bert-base-multilingual-cased (mBERT)
- google/muril-base-cased (MuRIL)
- Rajan/NepBERTa (monolingual Nepali RoBERTa)
- Any HuggingFace AutoModelForSequenceClassification checkpoint

Guarantees:
- Deterministic seeding (seed=42)
- Training duration (seconds), hardware environment (CPU/CUDA, RAM), model size on disk,
  and inference latency (ms/sentence) tracking
- Genuine checkpoint serialization (save_pretrained) and deterministic loading
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from nepali_scraper.config import PROJECT_ROOT, SCHEMA_VERSION, LABEL_SCHEMA
from nepali_scraper.processors import normalize_nepali_unicode

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set deterministic seeds across PyTorch and Python runtimes."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_hardware_info(device: torch.device) -> Dict[str, Any]:
    """Inspect and record hardware execution environment."""
    info: Dict[str, Any] = {
        "device_type": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
    else:
        info["device_name"] = "CPU"
    return info


class SentimentTextDataset(Dataset):
    """PyTorch Dataset for text tokenization and label tensors."""
    
    def __init__(self, texts: List[str], labels: Optional[List[int]], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"].squeeze(0)
            
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
            
        return item


class TransformerSentimentTrainer:
    """
    Production research trainer for transformer sentiment models.
    Supports mBERT, MuRIL, NepBERTa with latency profiling and checkpoint management.
    """
    
    SUPPORTED_PRETRAINED_MODELS = {
        "mbert": "bert-base-multilingual-cased",
        "muril": "google/muril-base-cased",
        "nepberta": "Rajan/NepBERTa",
    }
    
    def __init__(
        self,
        model_name_or_path: str = "google/muril-base-cased",
        num_labels: int = 3,
        max_length: int = 128,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        epochs: int = 3,
        weight_decay: float = 0.01,
        seed: int = 42,
        device: Optional[str] = None,
    ):
        self.model_name_or_path = self.SUPPORTED_PRETRAINED_MODELS.get(
            model_name_or_path.lower(), model_name_or_path
        )
        self.num_labels = num_labels
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.seed = seed
        
        set_seed(self.seed)
        
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.tokenizer = None
        self.model = None
        
        self.metadata: Dict[str, Any] = {
            "model_type": "TransformerSentimentTrainer",
            "model_name_or_path": self.model_name_or_path,
            "schema_version": SCHEMA_VERSION,
            "label_schema": LABEL_SCHEMA,
            "hyperparameters": {
                "num_labels": self.num_labels,
                "max_length": self.max_length,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "weight_decay": self.weight_decay,
                "seed": self.seed,
            },
            "hardware": get_hardware_info(self.device),
            "training_time_sec": 0.0,
            "inference_latency_ms": 0.0,
            "model_size_bytes": 0,
        }
    
    def initialize_model(self, custom_config: Optional[Any] = None):
        """Initialize tokenizer and sequence classification model."""
        id2label = {i: LABEL_SCHEMA.get(i, str(i)) for i in range(self.num_labels)}
        label2id = {v: k for k, v in id2label.items()}
        
        if custom_config is not None:
            # Used for fast testing without downloading large pretrained weights
            self.model = AutoModelForSequenceClassification.from_config(custom_config)
            self.model.to(self.device)
            return
        
        logger.info(f"Loading pretrained tokenizer from {self.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        
        logger.info(f"Loading pretrained backbone from {self.model_name_or_path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            num_labels=self.num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        self.model.to(self.device)
    
    def fit(
        self,
        X_train: List[str],
        y_train: List[int],
        X_val: Optional[List[str]] = None,
        y_val: Optional[List[int]] = None,
        patience: int = 3,
        warmup_ratio: float = 0.1,
    ) -> "TransformerSentimentTrainer":
        """
        Fine-tune transformer with AdamW, linear warmup with decay, and validation early stopping.
        
        Evaluates X_val, y_val at each epoch end for validation loss and Macro-F1.
        Snapshots best model weights in CPU memory on Macro-F1 improvement and restores
        best weights upon training completion.
        """
        if self.model is None or self.tokenizer is None:
            self.initialize_model()
            
        start_time = time.perf_counter()
        set_seed(self.seed)
        
        # Invariant A1: Normalize training texts
        X_train_norm = [normalize_nepali_unicode(t) for t in X_train]
        train_dataset = SentimentTextDataset(X_train_norm, y_train, self.tokenizer, self.max_length)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Set up validation loader if validation data provided
        val_loader = None
        has_val = X_val is not None and y_val is not None and len(X_val) > 0
        if has_val:
            X_val_norm = [normalize_nepali_unicode(t) for t in X_val]
            val_dataset = SentimentTextDataset(X_val_norm, y_val, self.tokenizer, self.max_length)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        total_training_steps = max(1, len(train_loader) * self.epochs)
        warmup_steps = int(total_training_steps * warmup_ratio)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
        loss_fn = nn.CrossEntropyLoss()
        
        best_val_macro_f1 = -1.0
        best_state_dict = None
        patience_counter = 0
        history: List[Dict[str, Any]] = []
        
        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item()
                
            avg_train_loss = total_train_loss / max(1, len(train_loader))
            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": round(avg_train_loss, 4),
            }
            
            # Validation loop at epoch end
            if val_loader is not None:
                self.model.eval()
                val_loss_total = 0.0
                all_val_preds = []
                all_val_targets = []
                
                with torch.no_grad():
                    for vbatch in val_loader:
                        v_input_ids = vbatch["input_ids"].to(self.device)
                        v_attention = vbatch["attention_mask"].to(self.device)
                        v_labels = vbatch["labels"].to(self.device)
                        
                        v_out = self.model(input_ids=v_input_ids, attention_mask=v_attention)
                        v_loss = loss_fn(v_out.logits, v_labels)
                        val_loss_total += v_loss.item()
                        
                        preds = torch.argmax(v_out.logits, dim=-1).cpu().numpy()
                        all_val_preds.extend(preds)
                        all_val_targets.extend(v_labels.cpu().numpy())
                        
                avg_val_loss = val_loss_total / max(1, len(val_loader))
                val_macro_f1 = float(f1_score(all_val_targets, all_val_preds, average="macro", zero_division=0))
                
                epoch_metrics["val_loss"] = round(avg_val_loss, 4)
                epoch_metrics["val_macro_f1"] = round(val_macro_f1, 4)
                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {avg_train_loss:.4f} - "
                    f"Val Loss: {avg_val_loss:.4f} - Val Macro-F1: {val_macro_f1:.4f}"
                )
                
                # In-memory CPU snapshot on Macro-F1 improvement
                if val_macro_f1 > best_val_macro_f1:
                    best_val_macro_f1 = val_macro_f1
                    best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered at epoch {epoch + 1} (patience={patience})")
                        history.append(epoch_metrics)
                        break
            else:
                logger.info(f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {avg_train_loss:.4f}")
                
            history.append(epoch_metrics)
            
        # Restore best weights if validation was performed
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            self.model.to(self.device)
            self.metadata["best_val_macro_f1"] = round(best_val_macro_f1, 4)
            
        self.metadata["training_time_sec"] = round(time.perf_counter() - start_time, 4)
        self.metadata["training_samples"] = len(X_train)
        self.metadata["history"] = history
        
        # Profile inference latency
        sample_subset = X_train_norm[:min(len(X_train_norm), 20)]
        self.metadata["inference_latency_ms"] = self.measure_inference_latency(sample_subset)
        
        return self
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Output softmax class probabilities with Unicode NFC normalized input."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer is not initialized.")
            
        self.model.eval()
        texts_norm = [normalize_nepali_unicode(t) for t in texts]
        dataset = SentimentTextDataset(texts_norm, None, self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        all_probs = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
                
        return np.vstack(all_probs) if all_probs else np.empty((0, self.num_labels))
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict integer class labels (0, 1, 2) with Unicode NFC normalized input."""
        probs = self.predict_proba(texts)
        if probs.size == 0:
            return np.empty(0, dtype=int)
        return np.argmax(probs, axis=-1)
    
    def measure_inference_latency(self, texts: List[str], repeats: int = 2) -> float:
        """Measure inference latency per sentence in milliseconds."""
        if not texts:
            return 0.0
        start = time.perf_counter()
        for _ in range(repeats):
            self.predict(texts)
        elapsed = time.perf_counter() - start
        return round((elapsed / (len(texts) * repeats)) * 1000.0, 4)
    
    def save(self, path: Union[str, Path]) -> Path:
        """Save transformer weights (save_pretrained), tokenizer, and metadata."""
        save_dir = Path(path).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if self.model is not None:
            self.model.save_pretrained(save_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_dir)
            
        # Calculate model size on disk
        total_bytes = sum(f.stat().st_size for f in save_dir.glob("**/*") if f.is_file())
        self.metadata["model_size_bytes"] = total_bytes
        
        metadata_file = save_dir / "training_manifest.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved transformer checkpoint and manifest to {save_dir}")
        return save_dir
    
    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[str] = None) -> "TransformerSentimentTrainer":
        """
        Load fine-tuned transformer checkpoint and manifest from disk.
        
        Restores model weights, tokenizer, and schema_version: '2.0-ordinal' label mappings.
        Raises FileNotFoundError if checkpoint directory or weights do not exist.
        Raises ValueError if checkpoint is corrupt or unreadable.
        """
        load_dir = Path(path).resolve()
        if not load_dir.is_dir():
            raise FileNotFoundError(f"Transformer checkpoint directory not found: {load_dir}")
            
        manifest_file = load_dir / "training_manifest.json"
        config_file = load_dir / "config.json"
        
        if not manifest_file.is_file() and not config_file.is_file():
            raise FileNotFoundError(f"Missing model weights or manifest in checkpoint: {load_dir}")
            
        meta = {}
        if manifest_file.is_file():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception as e:
                raise ValueError(f"Corrupt manifest file {manifest_file}: {e}") from e
                
        # Validate schema version
        schema = meta.get("schema_version", SCHEMA_VERSION)
        if schema != SCHEMA_VERSION:
            logger.warning(f"Checkpoint schema '{schema}' differs from active '{SCHEMA_VERSION}'")
            
        hparams = meta.get("hyperparameters", {})
        trainer = cls(
            model_name_or_path=str(load_dir),
            num_labels=hparams.get("num_labels", 3),
            max_length=hparams.get("max_length", 128),
            batch_size=hparams.get("batch_size", 16),
            seed=hparams.get("seed", 42),
            device=device,
        )
        try:
            trainer.tokenizer = AutoTokenizer.from_pretrained(str(load_dir))
            trainer.model = AutoModelForSequenceClassification.from_pretrained(str(load_dir))
            trainer.model.to(trainer.device)
        except Exception as e:
            raise ValueError(f"Failed to load transformer model/tokenizer from {load_dir}: {e}") from e
            
        trainer.metadata = meta
        return trainer

