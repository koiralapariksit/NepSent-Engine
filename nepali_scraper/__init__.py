"""
Nepali Sentiment Analysis - Web Scraping, Processing, Annotation & Benchmarking Package

Modules:
  - config: Centralized configuration and dynamic path resolution
  - processors: NepaliTextProcessor for Unicode NFC normalization, segmentation, and hashing
  - collectors: DatasetIngestor, provenance validator, and deduplication
  - annotation: Inter-Annotator Agreement (weighted Kappa) and human adjudication
  - splitter: Leakage-free, hash-grouped stratified dataset partitioner
  - scrapers: Domain-specific crawlers (news, e-commerce, social media)
  - utils: Logging, validation, and reporting utilities
"""

__version__ = "2.0.0"

from . import config
from . import processors
from . import collectors
from . import annotation
from . import splitter
from . import scrapers
from . import utils

from .processors import (
    NepaliTextProcessor,
    clean_zero_width_characters,
    compute_sentence_hash,
    devanagari_ratio,
    is_valid_nepali_text,
)

from .collectors import (
    DatasetIngestor,
    detect_hindi_contamination,
    char_ngram_jaccard,
)

from .annotation import (
    compute_iaa,
    export_for_annotation,
    adjudicate_annotations,
)

from .splitter import (
    hash_grouped_stratified_split,
    save_splits,
)

__all__ = [
    "config",
    "processors",
    "collectors",
    "annotation",
    "splitter",
    "scrapers",
    "utils",
    "NepaliTextProcessor",
    "clean_zero_width_characters",
    "compute_sentence_hash",
    "devanagari_ratio",
    "is_valid_nepali_text",
    "DatasetIngestor",
    "detect_hindi_contamination",
    "char_ngram_jaccard",
    "compute_iaa",
    "export_for_annotation",
    "adjudicate_annotations",
    "hash_grouped_stratified_split",
    "save_splits",
]
