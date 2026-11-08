"""
Unified Model Benchmark Training Orchestrator for NepSent-Engine.

CLI-driven runner that trains models across the 4-tier benchmark ladder:
- Level 0: Majority baseline
- Level 1: TF-IDF Logistic Regression & LinearSVC
- Level 2: Subword FastText equivalent
- Level 3 & 4: Fine-tuned transformers (mBERT, MuRIL, NepBERTa)

Saves checkpoints to `checkpoints/` and structured experiment run manifests to `runs/`.
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Any

# Ensure project root is in sys.path when executed directly
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

from nepali_scraper.config import PROJECT_ROOT, SCHEMA_VERSION, LABEL_SCHEMA
from nepali_scraper.models.classical import (
    MajorityBaselineClassifier,
    TfidfLinearModel,
    SubwordFastTextClassifier,
)

logger = logging.getLogger("nepali_scraper.benchmark")


def load_jsonl_partition(file_path: Path) -> Tuple[List[str], List[int], List[str]]:
    """Load sentences, labels, and domains from a JSONL partition."""
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Partition file not found: {path}")
        
    texts, labels, domains = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                rec = json.loads(line_str)
                t = rec.get("text") or rec.get("sentence")
                lbl = rec.get("label")
                dom = rec.get("domain", "unspecified")
                if t is not None and lbl is not None:
                    texts.append(str(t))
                    labels.append(int(lbl))
                    domains.append(str(dom))
                    
    return texts, labels, domains


def evaluate_predictions(y_true: List[int], y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute standard classification benchmark metrics."""
    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)
    
    macro_f1 = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)
    acc = accuracy_score(y_true_arr, y_pred_arr)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1, 2]).tolist()
    report = classification_report(y_true_arr, y_pred_arr, labels=[0, 1, 2], output_dict=True, zero_division=0)
    
    return {
        "macro_f1": round(float(macro_f1), 4),
        "micro_f1": round(float(micro_f1), 4),
        "accuracy": round(float(acc), 4),
        "confusion_matrix": cm,
        "per_class": {
            LABEL_SCHEMA[i]: {
                "precision": round(report[str(i)]["precision"], 4),
                "recall": round(report[str(i)]["recall"], 4),
                "f1": round(report[str(i)]["f1-score"], 4),
                "support": int(report[str(i)]["support"]),
            }
            for i in [0, 1, 2] if str(i) in report
        }
    }


def run_benchmark(
    data_dir: Path,
    model_choice: str = "all",
    output_dir: Optional[Path] = None,
    runs_dir: Optional[Path] = None,
    seed: int = 42,
    epochs: int = 3,
) -> Dict[str, Any]:
    """
    Train and evaluate selected models on verified partitions.
    """
    d_dir = Path(data_dir).resolve()
    chk_dir = Path(output_dir or PROJECT_ROOT / "checkpoints").resolve()
    r_dir = Path(runs_dir or PROJECT_ROOT / "runs").resolve()
    
    chk_dir.mkdir(parents=True, exist_ok=True)
    r_dir.mkdir(parents=True, exist_ok=True)
    
    # Locate train, val, and test partitions
    train_file = d_dir / "05_train_split.jsonl" if (d_dir / "05_train_split.jsonl").is_file() else d_dir / "train.jsonl"
    val_file = d_dir / "03_validation_partition.jsonl" if (d_dir / "03_validation_partition.jsonl").is_file() else d_dir / "val.jsonl"
    test_file = d_dir / "06_test_split.jsonl" if (d_dir / "06_test_split.jsonl").is_file() else d_dir / "test.jsonl"
    
    X_train, y_train, _ = load_jsonl_partition(train_file)
    X_val, y_val, _ = load_jsonl_partition(val_file) if val_file.is_file() else ([], [], [])
    X_test, y_test, _ = load_jsonl_partition(test_file)
    
    logger.info(f"Loaded partitions: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    model_choice = model_choice.lower()
    selected_models = []
    
    if model_choice in ["majority", "all"]:
        selected_models.append(("majority", MajorityBaselineClassifier(seed=seed)))
    if model_choice in ["tfidf_logreg", "all"]:
        selected_models.append(("tfidf_logreg", TfidfLinearModel(classifier_type="logistic_regression", seed=seed)))
    if model_choice in ["tfidf_svm", "all"]:
        selected_models.append(("tfidf_svm", TfidfLinearModel(classifier_type="linear_svc", seed=seed)))
    if model_choice in ["fasttext", "all"]:
        selected_models.append(("fasttext", SubwordFastTextClassifier(seed=seed)))
        
    benchmark_results: Dict[str, Any] = {}
    
    for name, model in selected_models:
        logger.info(f"--> Training {name}...")
        model.fit(X_train, y_train)
        
        # Save real checkpoint
        model_save_path = chk_dir / name
        model.save(model_save_path)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred)
        
        run_manifest = {
            "model_name": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "checkpoint_path": str(model_save_path),
            "test_metrics": metrics,
            "model_metadata": getattr(model, "metadata", {}),
        }
        
        # Write experiment manifest to runs/
        manifest_path = r_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{name}.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(run_manifest, f, indent=2, ensure_ascii=False)
            
        benchmark_results[name] = run_manifest
        logger.info(f"✓ {name} completed: Macro F1 = {metrics['macro_f1']:.4f}, Saved to {model_save_path}")
        
    return benchmark_results


def main():
    parser = argparse.ArgumentParser(description="NepSent-Engine Benchmark Runner")
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data"), help="Dataset directory")
    parser.add_argument("--model", type=str, default="all", choices=["majority", "tfidf_logreg", "tfidf_svm", "fasttext", "all"])
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "checkpoints"), help="Checkpoints directory")
    parser.add_argument("--runs-dir", type=str, default=str(PROJECT_ROOT / "runs"), help="Experiment runs directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=3, help="Transformer fine-tuning epochs")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    run_benchmark(
        data_dir=Path(args.data_dir),
        model_choice=args.model,
        output_dir=Path(args.output_dir),
        runs_dir=Path(args.runs_dir),
        seed=args.seed,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()

