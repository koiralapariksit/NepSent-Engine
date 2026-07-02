"""
Phase 3: Base Model Trainer Class

Provides unified interface for:
- Train/test/validation splits
- Cross-validation
- Evaluation metrics
- Model checkpointing
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    accuracy_score
)


class BaseModelTrainer:
    """Base trainer for all sentiment models."""
    
    def __init__(self, annotated_dataset_file: Path, model_name: str):
        """
        Initialize trainer.
        
        Args:
            annotated_dataset_file: Path to 04_annotated_dataset.jsonl
            model_name: Name of model (e.g., 'TF-IDF', 'fastText', 'mBERT', 'MuRIL')
        """
        self.dataset_file = annotated_dataset_file
        self.model_name = model_name
        self.sentences = []
        self.labels = []
        self.domains = []
        self.load_dataset()
    
    def load_dataset(self):
        """Load annotated dataset."""
        if not self.dataset_file.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_file}")
        
        with open(self.dataset_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    self.sentences.append(record['sentence'])
                    self.labels.append(record['label'])
                    self.domains.append(record['domain'])
                except (json.JSONDecodeError, KeyError):
                    pass
        
        print(f"✓ Loaded {len(self.sentences)} annotated sentences")
        print(f"  Domains: {set(self.domains)}")
        print(f"  Label distribution: {self._get_label_dist()}")
    
    def _get_label_dist(self) -> Dict[str, int]:
        """Get label distribution."""
        dist = {0: 0, 1: 0, 2: 0}
        for label in self.labels:
            dist[label] += 1
        return {
            "Negative": dist[0],
            "Neutral": dist[2],
            "Positive": dist[1]
        }
    
    def split_train_test(self, test_size: float = 0.15, random_state: int = 42) -> Tuple[List, List, List, List]:
        """
        Split data into train and test with stratification.
        
        Args:
            test_size: Test set ratio
            random_state: Random seed
            
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.sentences,
            self.labels,
            test_size=test_size,
            stratify=self.labels,
            random_state=random_state
        )
        
        print(f"\n[Train/Test Split]")
        print(f"  Train: {len(X_train)} sentences")
        print(f"  Test:  {len(X_test)} sentences")
        print(f"  Train label dist: {self._get_dist(y_train)}")
        print(f"  Test label dist:  {self._get_dist(y_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def _get_dist(self, labels: List) -> Dict:
        """Get distribution for label list."""
        dist = {0: 0, 1: 0, 2: 0}
        for label in labels:
            dist[label] += 1
        return {
            "Negative": dist[0],
            "Neutral": dist[2],
            "Positive": dist[1]
        }
    
    def cross_validate(self, X: List, y: List, n_splits: int = 5) -> List[Dict]:
        """
        Perform k-fold cross-validation.
        
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement cross_validate()")
    
    def evaluate(self, y_true: List, y_pred: List) -> Dict:
        """
        Compute evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with metrics
        """
        label_names = {0: "Negative", 1: "Positive", 2: "Neutral"}
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2], zero_division=0
        )
        
        report = classification_report(y_true, y_pred, labels=[0, 1, 2],
                                      target_names=[label_names[l] for l in [0, 1, 2]],
                                      zero_division=0)
        
        return {
            "accuracy": float(accuracy),
            "macro_f1": float(f1),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "confusion_matrix": cm.tolist(),
            "per_class": {
                label_names[i]: {
                    "precision": float(precision_per_class[i]),
                    "recall": float(recall_per_class[i]),
                    "f1": float(f1_per_class[i]),
                    "support": int(support[i])
                }
                for i in range(3)
            },
            "classification_report": report
        }
    
    def cross_domain_evaluation(self, X_train: List, y_train: List, 
                               domains_train: List, domains_test: List,
                               X_test: List, y_test: List) -> Dict:
        """
        Evaluate cross-domain transfer.
        
        Train on news_politics, evaluate on e-commerce + social_media.
        
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement cross_domain_evaluation()")
    
    def save_results(self, results: Dict, output_file: Path):
        """Save evaluation results."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Results saved: {output_file}")


class ModelBenchmark:
    """Benchmark multiple models."""
    
    def __init__(self, results_dir: Path):
        """Initialize benchmark."""
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_results = {}
    
    def add_model_results(self, model_name: str, results: Dict):
        """Add results for a model."""
        self.models_results[model_name] = results
    
    def generate_comparison(self) -> Dict:
        """Generate comparison across all models."""
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "models": {}
        }
        
        for model_name, results in self.models_results.items():
            comparison["models"][model_name] = {
                "accuracy": results.get("accuracy"),
                "macro_f1": results.get("macro_f1"),
                "macro_precision": results.get("macro_precision"),
                "macro_recall": results.get("macro_recall")
            }
        
        # Rank by F1
        ranked = sorted(
            comparison["models"].items(),
            key=lambda x: x[1]["macro_f1"],
            reverse=True
        )
        
        comparison["ranking"] = [
            {"rank": i+1, "model": name, "macro_f1": metrics["macro_f1"]}
            for i, (name, metrics) in enumerate(ranked)
        ]
        
        return comparison
    
    def save_comparison(self, comparison: Dict):
        """Save comparison results."""
        output_file = self.results_dir / "09_model_comparison.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Comparison saved: {output_file}")
        
        # Display ranking
        print("\n" + "="*80)
        print(" MODEL RANKING")
        print("="*80)
        print()
        for item in comparison["ranking"]:
            print(f"  {item['rank']}. {item['model']:20s} F1 = {item['macro_f1']:.4f}")
        print()
