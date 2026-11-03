"""
Configuration module for Nepali sentiment analysis scraper.
Centralized settings for all scraping, processing, and data management operations.
"""

from pathlib import Path

# ============================================================================
# ROOT & DIRECTORY PATHS (Dynamically resolved, no hardcoded machine paths)
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
TESTS_DIR = PROJECT_ROOT / "tests"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Ensure core runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA COLLECTION TARGETS
# ============================================================================

# Domain-specific target collection sizes (raw, pre-cleaning)
DOMAIN_TARGETS = {
    "news_politics": {
        "raw_target": 5000,
        "platforms": ["ratopati", "setopati", "onlinekhabar"],
        "description": "News & Politics domain"
    },
    "ecommerce_reviews": {
        "raw_target": 5000,
        "platforms": ["daraz"],
        "description": "E-commerce Product Reviews"
    },
    "social_media": {
        "raw_target": 4000,
        "platforms": ["twitter", "reddit"],
        "description": "Social & Conversational Media"
    }
}

# Empirical dataset targets
PILOT_COLLECTION_TARGET = 100     # 50-100 genuine records for pilot verification
INITIAL_DATASET_TARGET = 1000     # Minimum viable benchmark target
SCALED_DATASET_TARGET = 2000      # Target for balanced cross-domain benchmarking
MIN_PER_CLASS_TEST_SUPPORT = 80   # Statistical power requirement per class in test set

# ============================================================================
# TEXT PROCESSING PARAMETERS
# ============================================================================

# Unicode normalization form
UNICODE_NORM_FORM = "NFC"

# Devanagari script range (U+0900–U+097F)
DEVANAGARI_CHAR_PATTERN = r'[\u0900-\u097F]'

# Sentence tokenization delimiters (Nepali punctuation & standard sentence endings)
# Purna Viram (।), Question mark (?), Exclamation mark (!), Double danda (॥)
SENTENCE_DELIMITERS = r'([।?!॥\n])'

# Sentence length constraints (characters)
MIN_SENTENCE_LENGTH = 15   # Minimum meaningful content
MAX_SENTENCE_LENGTH = 350  # Maximum practical for transformer sequence classification

# Minimum proportion of Devanagari characters for valid Nepali sentence
MIN_DEVANAGARI_RATIO = 0.30  # Allows code-mixed English product names/terms

# ============================================================================
# WEB SCRAPING PARAMETERS
# ============================================================================

# HTTP request headers (polite crawling)
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 (NepSent Research Bot)"
    ),
    "Accept-Language": "ne,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Rate limiting (seconds between requests)
REQUEST_DELAY_BASE = 2.0        # Base delay between requests
REQUEST_DELAY_RANDOM_MAX = 3.0  # Add random 0-3 seconds
REQUEST_TIMEOUT = 10            # Connection timeout in seconds
MAX_RETRIES = 3                 # Max retry attempts for failed requests

# Platform-specific configurations
PLATFORM_CONFIG = {
    "ratopati": {
        "base_url": "https://ratopati.com",
        "rss_url": "https://ratopati.com/feed",
        "article_selector": "article",
        "text_selector": "p",
        "pagination_strategy": "url_param",
        "max_pages": 15,
    },
    "setopati": {
        "base_url": "https://setopati.com",
        "article_selector": "article",
        "text_selector": "p",
        "pagination_strategy": "url_param",
        "max_pages": 15,
    },
    "onlinekhabar": {
        "base_url": "https://onlinekhabar.com",
        "article_selector": "div.article-content",
        "text_selector": "p",
        "pagination_strategy": "url_param",
        "max_pages": 15,
    },
    "daraz": {
        "base_url": "https://www.daraz.com.np",
        "product_selector": "div.product",
        "review_selector": "div.review",
        "max_products_per_category": 500,
        "max_reviews_per_product": 5,
    },
    "twitter": {
        "search_query": "lang:ne",
        "batch_size": 100,
        "max_batches": 50,
        "note": "Requires Twitter API credentials (bearer token)"
    }
}

# ============================================================================
# ANNOTATION & VALIDATION PARAMETERS (Schema Version 2.0 Ordinal)
# ============================================================================

# Active schema: Ordinal 3-class (Negative < Neutral < Positive)
# Required for mathematically sound weighted Kappa (kappa_w) calculations
SCHEMA_VERSION = "2.0-ordinal"
LABEL_SCHEMA = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}
ID_TO_LABEL = LABEL_SCHEMA
LABEL_TO_ID = {v: k for k, v in LABEL_SCHEMA.items()}

# Legacy schema kept read-only for historical synthetic artifact verification
LEGACY_SCHEMA_VERSION = "1.0-nominal"
LEGACY_LABEL_SCHEMA = {
    0: "Negative",
    1: "Positive",
    2: "Neutral"
}

# Validation partition split (20% for dual-annotator overlap)
VALIDATION_PARTITION_RATIO = 0.20
ANNOTATOR_OVERLAP_RATIO = 0.20

# Cohen's Kappa thresholds for quality gates
KAPPA_THRESHOLDS = {
    "poor": 0.4,
    "moderate": 0.6,
    "substantial": 0.8,
    "perfect": 1.0
}

# ============================================================================
# OUTPUT FILE NAMING CONVENTIONS
# ============================================================================

OUTPUT_FILES = {
    "raw_corpus": DATA_DIR / "01_raw_corpus.jsonl",
    "cleaned_corpus": DATA_DIR / "02_cleaned_corpus.jsonl",
    "validation_partition": DATA_DIR / "03_validation_partition.jsonl",
    "annotated_dataset": DATA_DIR / "04_annotated_dataset.jsonl",
    "train_split": DATA_DIR / "05_train_split.jsonl",
    "test_split": DATA_DIR / "06_test_split.jsonl",
    "kappa_report": DATA_DIR / "07_kappa_report.json",
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"
LOG_FILE = LOGS_DIR / "scraper.log"

# ============================================================================
# DEBUGGING & DEVELOPMENT
# ============================================================================

DEBUG_MODE = False
SAMPLE_MODE = False
SAMPLE_SIZE = 10
