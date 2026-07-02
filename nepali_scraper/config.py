"""
Configuration module for Nepali sentiment analysis scraper.
Centralized settings for all scraping, processing, and data management operations.
"""

import os
from pathlib import Path

# Root paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ============================================================================
# DATA COLLECTION TARGETS
# ============================================================================

# Domain-specific target collection sizes (raw, pre-cleaning)
# Account for 50-60% data loss from normalization, deduplication, filtering
DOMAIN_TARGETS = {
    "news_politics": {
        "raw_target": 5000,  # Will yield ~2,000-2,500 cleaned
        "platforms": ["ratopati", "setopati", "onlinekhabar"],
        "description": "News & Politics domain"
    },
    "ecommerce_reviews": {
        "raw_target": 5000,  # Will yield ~2,000-2,500 cleaned
        "platforms": ["daraz"],
        "description": "E-commerce Product Reviews"
    },
    "social_media": {
        "raw_target": 4000,  # Will yield ~1,600-2,000 cleaned (Twitter/X data is noisier)
        "platforms": ["twitter"],
        "description": "Social & Conversational Media (Twitter/X)"
    }
}

# Expected final dataset size after processing
FINAL_DATASET_TARGET = 4000  # 4,000-5,000 verified sentences

# ============================================================================
# TEXT PROCESSING PARAMETERS
# ============================================================================

# Unicode normalization form
UNICODE_NORM_FORM = "NFC"

# Devanagari script range + ASCII basics + punctuation
# Includes Unicode range for Devanagari (U+0900–U+097F)
# Allows ASCII (U+0020–U+007E, includes digits and basic Latin)
# Allows newline and Nepali sentence delimiters
DEVANAGARI_SCRIPT_RANGE = r'([^\u0900-\u097F\u0020-\u007E\n।?!])'

# Sentence tokenization delimiters (Nepali punctuation)
SENTENCE_DELIMITERS = r'([।?!])'

# Sentence length constraints (characters)
MIN_SENTENCE_LENGTH = 15  # Minimum meaningful content
MAX_SENTENCE_LENGTH = 350  # Maximum practical for classification

# ============================================================================
# WEB SCRAPING PARAMETERS
# ============================================================================

# HTTP request headers (polite crawling)
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    ),
    "Accept-Language": "ne,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Rate limiting (seconds between requests)
REQUEST_DELAY_BASE = 2.0  # Base delay between requests
REQUEST_DELAY_RANDOM_MAX = 3.0  # Add random 0-3 seconds
REQUEST_TIMEOUT = 10  # Connection timeout in seconds
MAX_RETRIES = 3  # Max retry attempts for failed requests

# Platform-specific configurations
PLATFORM_CONFIG = {
    "ratopati": {
        "base_url": "https://ratopati.com",
        "article_selector": "article",
        "text_selector": "p",
        "pagination_strategy": "url_param",  # 'url_param', 'next_link', 'scroll'
        "max_pages": 15,  # ~300-400 articles
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
        "max_batches": 50,  # ~5,000 tweets if using official API
        "note": "Requires Twitter API credentials (bearer token)"
    }
}

# ============================================================================
# ANNOTATION & VALIDATION PARAMETERS
# ============================================================================

# Label schema (3 classes)
LABEL_SCHEMA = {
    0: "Negative",
    1: "Positive",
    2: "Neutral"
}

# Validation partition split (20% for dual-annotator overlap)
VALIDATION_PARTITION_RATIO = 0.20
ANNOTATOR_OVERLAP_RATIO = 0.20  # 1,000 sentences for overlap

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

DEBUG_MODE = False  # Set to True to print detailed debug info
SAMPLE_MODE = False  # Set to True to limit scraping (10 items per domain)
SAMPLE_SIZE = 10

print(f"[CONFIG] Project root: {PROJECT_ROOT}")
print(f"[CONFIG] Data directory: {DATA_DIR}")
print(f"[CONFIG] Logs directory: {LOGS_DIR}")
