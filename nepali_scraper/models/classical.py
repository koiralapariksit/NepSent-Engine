"""
Classical sentiment analysis baselines for NepSent-Engine.

Provides:
- Level 0: MajorityBaselineClassifier (predicts empirical majority class)
- Level 1: TfidfLinearModel (LogisticRegression & Calibrated LinearSVC with word/char n-grams)
- Level 2: SubwordFastTextClassifier (Subword char n-grams 3-6 bag model)

All models implement:
- fit(X, y)
- predict(X)
- predict_proba(X)
- save(path)
- load(path)
- Deterministic seeding (seed=42)
- Training duration (seconds), model size on disk (bytes), and inference latency (ms/sample) tracking.
"""

import time
import json
import logging
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple, Optional, Any, Union

import numpy as np
import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from nepali_scraper.config import PROJECT_ROOT, SCHEMA_VERSION, LABEL_SCHEMA

logger = logging.getLogger(__name__)


def _measure_inference_latency_ms(predict_fn, sample_texts: List[str], repeats: int = 3) -> float:
    """Measure average inference latency in milliseconds per sample."""
    if not sample_texts:
        return 0.0
    start = time.perf_counter()
    for _ in range(repeats):
        predict_fn(sample_texts)
    elapsed = time.perf_counter() - start
    return round((elapsed / (len(sample_texts) * repeats)) * 1000.0, 4)


class MajorityBaselineClassifier:
    """
    Level 0: Majority class baseline.
    Predicts the most frequent class in the training split.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.majority_class_: Optional[int] = None
        self.class_priors_: Dict[int, float] = {}
        self.classes_: List[int] = [0, 1, 2]
        self.metadata: Dict[str, Any] = {
            "model_type": "MajorityBaselineClassifier",
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "training_time_sec": 0.0,
            "inference_latency_ms": 0.0,
            "model_size_bytes": 0,
        }
    
    def fit(self, X: List[str], y: List[int]) -> "MajorityBaselineClassifier":
        """Fit majority baseline on training split."""
        start_time = time.perf_counter()
        
        y_int = [int(val) for val in y]
        counts = Counter(y_int)
        total = len(y_int)
        
        if total == 0:
            raise ValueError("Cannot fit on empty training set.")
        
        self.majority_class_ = counts.most_common(1)[0][0]
        self.class_priors_ = {c: counts.get(c, 0) / total for c in self.classes_}
        
        self.metadata["training_time_sec"] = round(time.perf_counter() - start_time, 4)
        self.metadata["majority_class"] = self.majority_class_
        self.metadata["class_priors"] = self.class_priors_
        self.metadata["training_samples"] = total
        
        # Measure baseline inference latency
        subset = X[:min(len(X), 50)]
        self.metadata["inference_latency_ms"] = _measure_inference_latency_ms(self.predict, subset)
        
        return self
    
    def predict(self, X: List[str]) -> np.ndarray:
        """Predict majority class for all inputs."""
        if self.majority_class_ is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")
        return np.full(len(X), self.majority_class_, dtype=int)
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Output empirical training class priors as probability distribution."""
        if not self.class_priors_:
            raise RuntimeError("Model is not fitted. Call fit() before predict_proba().")
        probs = [self.class_priors_.get(c, 0.0) for c in self.classes_]
        return np.tile(probs, (len(X), 1))
    
    def save(self, path: Union[str, Path]) -> Path:
        """Save model parameters and metadata JSON to disk."""
        save_dir = Path(path).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        
        weights_file = save_dir / "majority_model.joblib"
        metadata_file = save_dir / "metadata.json"
        
        state = {
            "majority_class_": self.majority_class_,
            "class_priors_": self.class_priors_,
            "classes_": self.classes_,
            "seed": self.seed,
        }
        joblib.dump(state, weights_file)
        
        self.metadata["model_size_bytes"] = weights_file.stat().st_size
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        return save_dir
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "MajorityBaselineClassifier":
        """Load saved checkpoint and metadata."""
        load_dir = Path(path).resolve()
        weights_file = load_dir / "majority_model.joblib"
        metadata_file = load_dir / "metadata.json"
        
        if not weights_file.is_file():
            raise FileNotFoundError(f"Missing weights file: {weights_file}")
        
        state = joblib.load(weights_file)
        model = cls(seed=state.get("seed", 42))
        model.majority_class_ = state["majority_class_"]
        model.class_priors_ = state["class_priors_"]
        model.classes_ = state["classes_"]
        
        if metadata_file.is_file():
            with open(metadata_file, "r", encoding="utf-8") as f:
                model.metadata = json.load(f)
        
        return model


class TfidfLinearModel:
    """
    Level 1: TF-IDF feature pipeline with Logistic Regression or Calibrated LinearSVC.
    
    Combines word n-grams (1, 2) and character n-grams (2, 4) tailored for Devanagari morphology.
    """
    
    def __init__(
        self,
        classifier_type: str = "logistic_regression",
        class_weight: Optional[str] = "balanced",
        C: float = 1.0,
        seed: int = 42,
        use_char_wb: bool = True,
        cv: int = 3,
    ):
        self.classifier_type = classifier_type.lower()
        self.class_weight = class_weight
        self.C = C
        self.seed = seed
        self.use_char_wb = use_char_wb
        self.cv = cv
        self.classes_: List[int] = [0, 1, 2]
        
        self.pipeline: Optional[Pipeline] = None
        self.metadata: Dict[str, Any] = {
            "model_type": f"TfidfLinearModel_{self.classifier_type}",
            "schema_version": SCHEMA_VERSION,
            "classifier_type": self.classifier_type,
            "class_weight": self.class_weight,
            "C": self.C,
            "seed": self.seed,
            "use_char_wb": self.use_char_wb,
            "cv": self.cv,
            "training_time_sec": 0.0,
            "inference_latency_ms": 0.0,
            "model_size_bytes": 0,
        }
    
    def _build_pipeline(self, cv_folds: Optional[int] = None) -> Pipeline:
        """Construct the feature extraction and classification pipeline."""
        if self.use_char_wb:
            # Combined Word + Char-WB features tailored for Devanagari inflections
            features = FeatureUnion([
                ("word_tfidf", TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                )),
                ("char_tfidf", TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 4),
                    sublinear_tf=True,
                    min_df=1,
                )),
            ])
        else:
            features = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 3),
                sublinear_tf=True,
                min_df=1,
            )
        
        if self.classifier_type == "logistic_regression":
            clf = LogisticRegression(
                C=self.C,
                max_iter=1000,
                class_weight=self.class_weight,
                random_state=self.seed,
                solver="lbfgs",
            )
        elif self.classifier_type in ["linear_svc", "svc", "linearsvc"]:
            base_svc = LinearSVC(
                C=self.C,
                max_iter=2000,
                class_weight=self.class_weight,
                random_state=self.seed,
            )
            # Calibrate SVM to provide well-behaved probabilities
            actual_cv = cv_folds if cv_folds is not None else self.cv
            if actual_cv and actual_cv >= 2:
                clf = CalibratedClassifierCV(estimator=base_svc, cv=actual_cv)
            else:
                clf = base_svc
        else:
            raise ValueError(f"Unsupported classifier_type: {self.classifier_type}")
        
        return Pipeline([
            ("features", features),
            ("classifier", clf),
        ])
    
    def fit(self, X: List[str], y: List[int]) -> "TfidfLinearModel":
        """Fit TF-IDF pipeline on training texts and labels."""
        start_time = time.perf_counter()
        
        y_int = np.array([int(val) for val in y], dtype=int)
        
        # Adapt CV to minimum per-class samples if calibrating LinearSVC
        counts = Counter(y_int)
        min_class_count = min(counts.values()) if counts else 0
        actual_cv = min(self.cv, min_class_count) if min_class_count >= 2 else None
        
        self.pipeline = self._build_pipeline(cv_folds=actual_cv)
        self.pipeline.fit(X, y_int)
        
        self.metadata["training_time_sec"] = round(time.perf_counter() - start_time, 4)
        self.metadata["training_samples"] = len(X)
        self.metadata["effective_cv"] = actual_cv
        
        # Measure inference latency
        subset = X[:min(len(X), 50)]
        self.metadata["inference_latency_ms"] = _measure_inference_latency_ms(self.predict, subset)
        
        return self
    
    def predict(self, X: List[str]) -> np.ndarray:
        """Predict labels for input texts."""
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Predict class probabilities."""
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict_proba().")
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X)
        
        # Calibrated fallback if base LinearSVC was used without CalibratedClassifierCV
        decision = self.pipeline.decision_function(X)
        if decision.ndim == 1:
            decision = np.vstack([-decision, decision]).T
        exp_d = np.exp(decision - np.max(decision, axis=1, keepdims=True))
        return exp_d / exp_d.sum(axis=1, keepdims=True)
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict_proba().")
        return self.pipeline.predict_proba(X)
    
    def save(self, path: Union[str, Path]) -> Path:
        """Save pipeline and metadata JSON to disk."""
        save_dir = Path(path).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        
        weights_file = save_dir / "tfidf_model.joblib"
        metadata_file = save_dir / "metadata.json"
        
        joblib.dump(self.pipeline, weights_file)
        self.metadata["model_size_bytes"] = weights_file.stat().st_size
        
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        return save_dir
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "TfidfLinearModel":
        """Load saved checkpoint and metadata."""
        load_dir = Path(path).resolve()
        weights_file = load_dir / "tfidf_model.joblib"
        metadata_file = load_dir / "metadata.json"
        
        if not weights_file.is_file():
            raise FileNotFoundError(f"Missing weights file: {weights_file}")
        
        meta = {}
        if metadata_file.is_file():
            with open(metadata_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        
        model = cls(
            classifier_type=meta.get("classifier_type", "logistic_regression"),
            class_weight=meta.get("class_weight", "balanced"),
            C=meta.get("C", 1.0),
            seed=meta.get("seed", 42),
            use_char_wb=meta.get("use_char_wb", True),
        )
        model.pipeline = joblib.load(weights_file)
        model.metadata = meta
        return model


class SubwordFastTextClassifier:
    """
    Level 2: Subword Character N-Gram Classifier (fastText architecture equivalent).
    
    Extracts character 3-6 n-grams into a sublinear bag-of-ngrams representation
    with L2 regularized logistic regression, running reliably across Windows and Linux
    without native C++ compilation dependencies.
    """
    
    def __init__(
        self,
        min_ngram: int = 3,
        max_ngram: int = 6,
        C: float = 1.0,
        class_weight: Optional[str] = "balanced",
        seed: int = 42,
    ):
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram
        self.C = C
        self.class_weight = class_weight
        self.seed = seed
        self.pipeline: Optional[Pipeline] = None
        
        self.metadata: Dict[str, Any] = {
            "model_type": "SubwordFastTextClassifier",
            "schema_version": SCHEMA_VERSION,
            "min_ngram": min_ngram,
            "max_ngram": max_ngram,
            "C": C,
            "class_weight": class_weight,
            "seed": seed,
            "training_time_sec": 0.0,
            "inference_latency_ms": 0.0,
            "model_size_bytes": 0,
        }
    
    def fit(self, X: List[str], y: List[int]) -> "SubwordFastTextClassifier":
        """Fit subword n-gram fastText pipeline."""
        start_time = time.perf_counter()
        
        y_int = np.array([int(val) for val in y], dtype=int)
        
        # fastText subword formulation: sublinear TF-IDF of character n-grams (3 to 6)
        subword_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(self.min_ngram, self.max_ngram),
            sublinear_tf=True,
            min_df=1,
        )
        
        classifier = LogisticRegression(
            C=self.C,
            max_iter=1000,
            class_weight=self.class_weight,
            random_state=self.seed,
            solver="lbfgs",
        )
        
        self.pipeline = Pipeline([
            ("subwords", subword_vectorizer),
            ("classifier", classifier),
        ])
        
        self.pipeline.fit(X, y_int)
        
        self.metadata["training_time_sec"] = round(time.perf_counter() - start_time, 4)
        self.metadata["training_samples"] = len(X)
        self.metadata["vocabulary_size"] = len(subword_vectorizer.vocabulary_)
        
        # Measure inference latency
        subset = X[:min(len(X), 50)]
        self.metadata["inference_latency_ms"] = _measure_inference_latency_ms(self.predict, subset)
        
        return self
    
    def predict(self, X: List[str]) -> np.ndarray:
        """Predict labels using subword representations."""
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Predict class probabilities."""
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict_proba().")
        return self.pipeline.predict_proba(X)
    
    def save(self, path: Union[str, Path]) -> Path:
        """Save model checkpoint and metadata JSON to disk."""
        save_dir = Path(path).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        
        weights_file = save_dir / "fasttext_model.joblib"
        metadata_file = save_dir / "metadata.json"
        
        joblib.dump(self.pipeline, weights_file)
        self.metadata["model_size_bytes"] = weights_file.stat().st_size
        
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        return save_dir
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "SubwordFastTextClassifier":
        """Load saved checkpoint and metadata."""
        load_dir = Path(path).resolve()
        weights_file = load_dir / "fasttext_model.joblib"
        metadata_file = load_dir / "metadata.json"
        
        if not weights_file.is_file():
            raise FileNotFoundError(f"Missing weights file: {weights_file}")
        
        meta = {}
        if metadata_file.is_file():
            with open(metadata_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        
        model = cls(
            min_ngram=meta.get("min_ngram", 3),
            max_ngram=meta.get("max_ngram", 6),
            C=meta.get("C", 1.0),
            class_weight=meta.get("class_weight", "balanced"),
            seed=meta.get("seed", 42),
        )
        model.pipeline = joblib.load(weights_file)
        model.metadata = meta
        return model

