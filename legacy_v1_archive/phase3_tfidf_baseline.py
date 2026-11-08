"""
Phase 3 - Level 1: TF-IDF + Logistic Regression Baseline

Simplest model in the 4-tier ladder.
Target: 55-65% Macro F1
Features: Char-level trigrams
Classifier: Logistic Regression
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from phase3_model_trainer import BaseModelTrainer


class TFIDFModel(BaseModelTrainer):
    """TF-IDF + Logistic Regression."""
    
    def __init__(self, annotated_dataset_file: Path):
        """Initialize TF-IDF model."""
        super().__init__(annotated_dataset_file, "TF-IDF")
        self.vectorizer = None
        self.classifier = None
    
    def train(self, X_train: List, y_train: List):
        """
        Train TF-IDF model.
        
        Args:
            X_train: Training sentences
            y_train: Training labels
        """
        print("\n" + "="*80)
        print(" LEVEL 1: TF-IDF + LOGISTIC REGRESSION")
        print("="*80)
        print()
        
        # TF-IDF Vectorizer with char-level trigrams
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 3),  # Char trigrams only
            max_features=2000,
            min_df=1,
            max_df=0.95,
            lowercase=True
        )
        
        print("[Training]")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        print(f"  TF-IDF matrix shape: {X_train_tfidf.shape}")
        print(f"  Features: {X_train_tfidf.shape[1]} char-trigrams")
        
        # Logistic Regression classifier
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        
        self.classifier.fit(X_train_tfidf, y_train)
        print(f"  Logistic Regression trained")
        print()
        
        return self
    
    def predict(self, X: List) -> List:
        """Predict labels."""
        if self.vectorizer is None or self.classifier is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_tfidf = self.vectorizer.transform(X)
        return self.classifier.predict(X_tfidf)
    
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
            
            # Train on this fold
            vectorizer = TfidfVectorizer(
                analyzer='char',
                ngram_range=(3, 3),
                max_features=2000,
                min_df=1,
                max_df=0.95,
                lowercase=True
            )
            X_train_tfidf = vectorizer.fit_transform(X_train_fold)
            
            classifier = LogisticRegression(max_iter=1000, random_state=42)
            classifier.fit(X_train_tfidf, y_train_fold)
            
            # Evaluate on validation fold
            X_val_tfidf = vectorizer.transform(X_val_fold)
            y_pred = classifier.predict(X_val_tfidf)
            
            fold_metrics = self.evaluate(y_val_fold, y_pred)
            fold_results.append(fold_metrics)
            
            print(f"  Fold {fold+1}: F1 = {fold_metrics['macro_f1']:.4f}")
        
        # Aggregate results
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
        
        # Filter to news domain for training
        X_train_news = []
        y_train_news = []
        for i, domain in enumerate(domains_train):
            if domain == "news_politics":
                X_train_news.append(X_train[i])
                y_train_news.append(y_train[i])
        
        # Filter to non-news domains for testing
        X_test_other = []
        y_test_other = []
        for i, domain in enumerate(domains_test):
            if domain != "news_politics":
                X_test_other.append(X_test[i])
                y_test_other.append(y_test[i])
        
        print(f"  Train samples: {len(X_train_news)} (news)")
        print(f"  Test samples:  {len(X_test_other)} (e-commerce + social)")
        
        # Train on news only
        vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 3),
            max_features=2000,
            min_df=1,
            max_df=0.95,
            lowercase=True
        )
        X_train_tfidf = vectorizer.fit_transform(X_train_news)
        
        classifier = LogisticRegression(max_iter=1000, random_state=42)
        classifier.fit(X_train_tfidf, y_train_news)
        
        # Evaluate on other domains
        X_test_tfidf = vectorizer.transform(X_test_other)
        y_pred = classifier.predict(X_test_tfidf)
        
        results = self.evaluate(y_test_other, y_pred)
        
        print(f"  Macro F1 (cross-domain): {results['macro_f1']:.4f}")
        print(f"  (Baseline F1 on news: typically 0.65-0.75)")
        print()
        
        return results


def main():
    """Execute TF-IDF baseline training."""
    
    dataset_file = Path("d:/ds mdoel/data/04_annotated_dataset.jsonl")
    output_file = Path("d:/ds mdoel/data/08a_tfidf_results.json")
    
    # Initialize model
    model = TFIDFModel(dataset_file)
    
    # Split data
    X_train, X_test, y_train, y_test = model.split_train_test(test_size=0.15)
    
    # Train model
    model.train(X_train, y_train)
    
    # Evaluate on test set
    results = model.evaluate_on_test(X_test, y_test)
    
    # Cross-validation
    cv_results = model.cross_validate(X_train + X_test, y_train + y_test, n_splits=5)
    results["cross_validation"] = cv_results
    
    # Cross-domain evaluation
    domains_train = [model.domains[i] for i in range(len(X_train))]
    domains_test = [model.domains[len(X_train) + i] for i in range(len(X_test))]
    
    cd_results = model.cross_domain_evaluation(
        X_train, y_train, domains_train, domains_test,
        X_test, y_test
    )
    results["cross_domain"] = cd_results
    
    # Save results
    model.save_results(results, output_file)
    
    print("=" * 80)
    print(" TF-IDF BASELINE COMPLETE")
    print("=" * 80)
    print()
    print(f"Results: Macro F1 = {results['macro_f1']:.4f}")
    print(f"CV Folds Mean F1  = {cv_results['mean_macro_f1']:.4f}")
    print(f"Cross-Domain F1   = {cd_results['macro_f1']:.4f}")
    print()


if __name__ == "__main__":
    main()
