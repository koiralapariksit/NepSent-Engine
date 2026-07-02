"""
Phase 3 - Level 4: Fine-tuned MuRIL (Multilingual and Code-mixed BERT for Indian Languages)

Best model in the 4-tier ladder, optimized for Indic languages including Nepali.
Target: 79-86% Macro F1
Model: google/muril-base-cased
Training: 3 epochs, batch_size=32, lr=5e-5, seq_len=128
Advantage: Pre-trained on Nepali and other Indic languages specifically
"""

import json
import sys
import torch
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim import AdamW
except ImportError:
    print("Installing transformers...")
    import subprocess
    subprocess.check_call(["pip", "install", "transformers", "torch"])
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim import AdamW

from phase3_model_trainer import BaseModelTrainer


class MuRILModel(BaseModelTrainer):
    """Fine-tuned MuRIL (Indic-specific BERT)."""
    
    MODEL_NAME = "google/muril-base-cased"
    
    def __init__(self, annotated_dataset_file: Path):
        """Initialize MuRIL model."""
        super().__init__(annotated_dataset_file, "MuRIL")
        self.tokenizer = None
        self.model = None
        self.device = None
        self.load_model()
    
    def load_model(self):
        """Load pre-trained MuRIL."""
        print("\n[Loading MuRIL Model]")
        print(f"  Model: {self.MODEL_NAME}")
        print(f"  (Optimized for Indic languages including Nepali)")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.MODEL_NAME,
                num_labels=3  # 3 sentiment classes
            )
            self.model.to(self.device)
            print(f"  ✓ MuRIL model loaded successfully")
        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            print(f"  Ensure: pip install transformers torch")
            raise
    
    def tokenize(self, sentences: List, max_length: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize sentences with MuRIL tokenizer."""
        encoded = self.tokenizer(
            sentences,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        return encoded['input_ids'], encoded['attention_mask']
    
    def train(self, X_train: List, y_train: List, epochs: int = 3, batch_size: int = 32, lr: float = 5e-5):
        """
        Fine-tune MuRIL on sentiment data.
        
        Note: MuRIL uses higher learning rate (5e-5) compared to mBERT (2e-5)
        because it's already optimized for Indic languages.
        
        Args:
            X_train: Training sentences
            y_train: Training labels
            epochs: Number of epochs
            batch_size: Batch size
            lr: Learning rate (5e-5 recommended for MuRIL)
        """
        print("\n" + "="*80)
        print(" LEVEL 4: FINE-TUNED MURIL (BEST MODEL)")
        print("="*80)
        print()
        print(f"[Training Configuration]")
        print(f"  Model: google/muril-base-cased")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {lr}")
        print(f"  Max sequence length: 128")
        print(f"  (MuRIL is pre-trained on Indic languages including Nepali)")
        print()
        
        print("[Tokenizing training data]")
        input_ids, attention_mask = self.tokenize(X_train)
        labels = torch.tensor(y_train, dtype=torch.long)
        
        dataset = TensorDataset(input_ids, attention_mask, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"  Batches: {len(dataloader)}")
        print()
        
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=lr)
        
        # Training loop
        self.model.train()
        
        for epoch in range(epochs):
            print(f"[Epoch {epoch+1}/{epochs}]")
            total_loss = 0
            
            for batch_idx, (batch_input_ids, batch_attn_mask, batch_labels) in enumerate(tqdm(dataloader)):
                batch_input_ids = batch_input_ids.to(self.device)
                batch_attn_mask = batch_attn_mask.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attn_mask,
                    labels=batch_labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
            
            avg_loss = total_loss / len(dataloader)
            print(f"  Average loss: {avg_loss:.4f}")
        
        print()
        return self
    
    def predict(self, X: List) -> List:
        """Predict labels."""
        self.model.eval()
        
        with torch.no_grad():
            input_ids, attention_mask = self.tokenize(X)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            predictions = torch.argmax(outputs.logits, dim=1)
            return predictions.cpu().numpy()
    
    def evaluate_on_test(self, X_test: List, y_test: List) -> Dict:
        """Evaluate on test set."""
        print("[Evaluation on Test Set]")
        y_pred = self.predict(X_test)
        results = self.evaluate(y_test, y_pred)
        
        print(f"  Accuracy:    {results['accuracy']:.4f}")
        print(f"  Macro F1:    {results['macro_f1']:.4f}")
        print(f"  Precision:   {results['macro_precision']:.4f}")
        print(f"  Recall:      {results['macro_recall']:.4f}")
        print()
        
        return results
    
    def cross_validate(self, X: List, y: List, n_splits: int = 5) -> Dict:
        """Note: Full CV skipped for speed. Use single train/test."""
        return {
            "note": "Single train/test used for efficiency",
        }
    
    def cross_domain_evaluation(self, X_train: List, y_train: List,
                               domains_train: List, domains_test: List,
                               X_test: List, y_test: List) -> Dict:
        """Train on news, test on e-commerce + social_media."""
        print("\n[Cross-Domain Evaluation]")
        print("  Train: news_politics only")
        print("  Test: e-commerce + social_media only")
        
        # Filter
        X_train_news = []
        y_train_news = []
        for i, domain in enumerate(domains_train):
            if domain == "news_politics":
                X_train_news.append(X_train[i])
                y_train_news.append(y_train[i])
        
        domains_test_idx = list(range(len(X_test)))
        X_test_other = [X_test[i] for i in domains_test_idx if domains_test[i] != "news_politics"]
        y_test_other = [y_test[i] for i in domains_test_idx if domains_test[i] != "news_politics"]
        
        print(f"  Train samples: {len(X_train_news)}")
        print(f"  Test samples:  {len(X_test_other)}")
        
        if len(X_train_news) < 5 or len(X_test_other) < 3:
            print(f"  ✗ Insufficient samples")
            return {"error": "Not enough samples"}
        
        # Fine-tune on news
        print(f"  Fine-tuning on news domain...")
        self.train(X_train_news, y_train_news, epochs=1, batch_size=16, lr=5e-5)
        
        # Evaluate
        y_pred = self.predict(X_test_other)
        results = self.evaluate(y_test_other, y_pred)
        
        print(f"  Macro F1 (cross-domain): {results['macro_f1']:.4f}")
        print()
        
        return results


def main():
    """Execute MuRIL fine-tuning."""
    
    dataset_file = Path("d:/ds mdoel/data/04_annotated_dataset.jsonl")
    output_file = Path("d:/ds mdoel/data/08d_muril_results.json")
    
    # Initialize model
    model = MuRILModel(dataset_file)
    
    # Split data
    X_train, X_test, y_train, y_test = model.split_train_test(test_size=0.15)
    
    # Train model
    model.train(X_train, y_train, epochs=3, batch_size=32, lr=5e-5)
    
    # Evaluate
    results = model.evaluate_on_test(X_test, y_test)
    
    # Cross-domain
    domains_train = [model.domains[i] for i in range(len(X_train))]
    domains_test = [model.domains[len(X_train) + i] for i in range(len(X_test))]
    
    cd_results = model.cross_domain_evaluation(
        X_train, y_train, domains_train, domains_test,
        X_test, y_test
    )
    results["cross_domain"] = cd_results
    
    # Save
    model.save_results(results, output_file)
    
    print("=" * 80)
    print(" MURIL FINE-TUNING COMPLETE")
    print("=" * 80)
    print()
    print(f"Results: Macro F1 = {results['macro_f1']:.4f}")
    print(f"Cross-Domain F1   = {cd_results.get('macro_f1', 'N/A')}")
    print()


if __name__ == "__main__":
    main()
