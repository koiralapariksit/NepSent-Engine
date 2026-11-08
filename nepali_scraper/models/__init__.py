"""
Model Benchmarking Ladder for NepSent-Engine.

Provides:
- Level 0: MajorityBaselineClassifier (trivial baseline)
- Level 1: TfidfLinearModel (LogisticRegression & Calibrated LinearSVC)
- Level 2: SubwordFastTextClassifier (Subword n-gram fastText equivalent)
- Level 3 & 4: TransformerSentimentTrainer (mBERT, MuRIL, NepBERTa)
"""

from .classical import (
    MajorityBaselineClassifier,
    TfidfLinearModel,
    SubwordTfidfClassifier,
    SubwordFastTextClassifier,
)

from .transformers_Engine import (
    TransformerSentimentTrainer,
)

__all__ = [
    "MajorityBaselineClassifier",
    "TfidfLinearModel",
    "SubwordTfidfClassifier",
    "SubwordFastTextClassifier",
    "TransformerSentimentTrainer",
]
