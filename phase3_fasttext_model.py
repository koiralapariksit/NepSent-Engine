"""
Phase 3 - Level 2: fastText Embeddings + Logistic Regression

Uses pre-trained fastText embeddings for Nepali.
Target: 63-72% Macro F1
Features: 300-dimensional fastText embeddings
Classifier: Logistic Regression
"""

import json
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict
from urllib.request import urlretrieve
import zipfile
import os

sys.path.insert(0, str(Path(__file__).parent))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from phase3_model_trainer import BaseModelTrainer

try:
    import fasttext
except ImportError:
    print("Installing fasttext...")
    os.system("pip install fasttext")
    import fasttext


class FastTextModel(BaseModelTrainer):
    """fastText embeddings + Logistic Regression."""
    
    FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ne.300.bin.gz"
    FASTTEXT_MODEL_PATH = Path("d:/ds mdoel/models/cc.ne.300.bin")
    
    def __init__(self, annotated_dataset_file: Path):
        """Initialize fastText model."""
        super().__init__(annotated_dataset_file, "fastText")
        self.embedder = None
        self.classifier = None
        self.load_embedder()
    
    def load_embedder(self):
        """Load or download fastText model."""
        print("\n[Loading fastText Model]")
        
        if self.FASTTEXT_MODEL_PATH.exists():
            print(f"  Loading: {self.FASTTEXT_MODEL_PATH}")
            self.embedder = fasttext.load_model(str(self.FASTTEXT_MODEL_PATH))
            print(f"  ✓ Model loaded")
        else:
            print(f"  Downloading fastText model for Nepali (cc.ne.300.bin)...")
            print(f"  This only happens once and takes ~30 seconds")
            
            self.FASTTEXT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            # Download compressed model
            gz_path = str(self.FASTTEXT_MODEL_PATH) + ".gz"
            try:
                urlretrieve(self.FASTTEXT_MODEL_URL, gz_path)
                print(f"  ✓ Downloaded")
                
                # Extract
                import gzip
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(self.FASTTEXT_MODEL_PATH, 'wb') as f_out:
                        f_out.write(f_in.read())
                
                os.remove(gz_path)
                print(f"  ✓ Extracted")
            except Exception as e:
                print(f"  ✗ Download failed: {e}")
                print(f"  Using TF-IDF as fallback")
                return None
            
            self.embedder = fasttext.load_model(str(self.FASTTEXT_MODEL_PATH))
            print(f"  ✓ Model ready")
    
    def get_sentence_embedding(self, sentence: str) -> np.ndarray:
        """Get 300-dim embedding for sentence."""
        if self.embedder is None:
            raise ValueError("Embedder not loaded")
        
        # Average word embeddings
        words = sentence.lower().split()
        if not words:
            return np.zeros(300)
        
        embeddings = [self.embedder.get_word_vector(word) for word in words]
        return np.mean(embeddings, axis=0)
    
    def embed_sentences(self, sentences: List) -> np.ndarray:
        """Embed all sentences."""
        embeddings = []
        for i, sent in enumerate(sentences):
            if (i + 1) % max(1, len(sentences) // 10) == 0:
                print(f"    Embedding {i+1}/{len(sentences)}...", end='\r')
            embeddings.append(self.get_sentence_embedding(sent))
        print(f"    Embedded {len(sentences)} sentences" + " " * 30)
        return np.array(embeddings)
    
    def train(self, X_train: List, y_train: List):
        """
        Train fastText model (just embeddings + LogReg).
        
        Args:
            X_train: Training sentences
            y_train: Training labels
        """
        print("\n" + "="*80)
        print(" LEVEL 2: FASTTEXT EMBEDDINGS + LOGISTIC REGRESSION")
        print("="*80)
        print()
        
        print("[Training]")
        print("  Embedding sentences...")
        X_train_emb = self.embed_sentences(X_train)
        
        print(f"  Embedding matrix shape: {X_train_emb.shape}")
        print(f"  Features: 300-dimensional Nepali embeddings")
        
        # Logistic Regression on embeddings
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        
        self.classifier.fit(X_train_emb, y_train)
        print(f"  Logistic Regression trained")
        print()
        
        return self
    
    def predict(self, X: List) -> List:
        """Predict labels."""
        if self.classifier is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_emb = self.embed_sentences(X)
        return self.classifier.predict(X_emb)
    
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
        """Perform 5-fold cross-validation."""
        print(f"\n[{n_splits}-Fold Cross-Validation]")
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold = [X[i] for i in train_idx]
            y_train_fold = [y[i] for i in train_idx]
            X_val_fold = [X[i] for i in val_idx]
            y_val_fold = [y[i] for i in val_idx]
            
            # Embed and train
            print(f"  Fold {fold+1}: Embedding...", end='\r')
            X_train_emb = self.embed_sentences(X_train_fold)
            
            classifier = LogisticRegression(max_iter=1000, random_state=42)
            classifier.fit(X_train_emb, y_train_fold)
            
            # Evaluate
            X_val_emb = self.embed_sentences(X_val_fold)
            y_pred = classifier.predict(X_val_emb)
            
            fold_metrics = self.evaluate(y_val_fold, y_pred)
            fold_results.append(fold_metrics)
            
            print(f"  Fold {fold+1}: F1 = {fold_metrics['macro_f1']:.4f}" + " " * 20)
        
        # Aggregate
        mean_f1 = sum(r['macro_f1'] for r in fold_results) / len(fold_results)
        mean_accuracy = sum(r['accuracy'] for r in fold_results) / len(fold_results)
        mean_precision = sum(r['macro_precision'] for r in fold_results) / len(fold_results)
        mean_recall = sum(r['macro_recall'] for r in fold_results) / len(fold_results)
        
        print(f"\n  Mean Accuracy:    {mean_accuracy:.4f}")
        print(f"  Mean Macro F1:    {mean_f1:.4f}")
        print(f"  Mean Precision:   {mean_precision:.4f}")
        print(f"  Mean Recall:      {mean_recall:.4f}")
        print()
        
        return {
            "cv_folds": fold_results,
            "mean_accuracy": mean_accuracy,
            "mean_macro_f1": mean_f1,
            "mean_macro_precision": mean_precision,
            "mean_macro_recall": mean_recall
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
        
        X_test_other = []
        y_test_other = []
        for i, domain in enumerate(domains_test):
            if domain != "news_politics":
                X_test_other.append(X_test[i])
                y_test_other.append(y_test[i])
        
        print(f"  Train samples: {len(X_train_news)}")
        print(f"  Test samples:  {len(X_test_other)}")
        
        # Train and evaluate
        X_train_emb = self.embed_sentences(X_train_news)
        classifier = LogisticRegression(max_iter=1000, random_state=42)
        classifier.fit(X_train_emb, y_train_news)
        
        X_test_emb = self.embed_sentences(X_test_other)
        y_pred = classifier.predict(X_test_emb)
        
        results = self.evaluate(y_test_other, y_pred)
        print(f"  Macro F1 (cross-domain): {results['macro_f1']:.4f}")
        print()
        
        return results


def main():
    """Execute fastText model training."""
    
    dataset_file = Path("d:/ds mdoel/data/04_annotated_dataset.jsonl")
    output_file = Path("d:/ds mdoel/data/08b_fasttext_results.json")
    
    # Initialize model
    model = FastTextModel(dataset_file)
    
    if model.embedder is None:
        print("\n✗ fastText model failed to load. Skipping...")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = model.split_train_test(test_size=0.15)
    
    # Train model
    model.train(X_train, y_train)
    
    # Evaluate
    results = model.evaluate_on_test(X_test, y_test)
    
    # Cross-validation
    cv_results = model.cross_validate(X_train + X_test, y_train + y_test, n_splits=5)
    results["cross_validation"] = cv_results
    
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
    print(" FASTTEXT MODEL COMPLETE")
    print("=" * 80)
    print()
    print(f"Results: Macro F1 = {results['macro_f1']:.4f}")
    print(f"CV Folds Mean F1  = {cv_results['mean_macro_f1']:.4f}")
    print(f"Cross-Domain F1   = {cd_results['macro_f1']:.4f}")
    print()


if __name__ == "__main__":
    main()
