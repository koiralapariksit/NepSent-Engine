"""
Nepali Sentiment Analysis - Web Scraping Package

A production-grade multi-domain web scraping system for collecting 
4,000–5,000 high-quality Nepali sentences for sentiment analysis research.

Modules:
  - config: Centralized configuration for all scraping operations
  - processors: NepaliTextProcessor for Unicode normalization and segmentation
  - scrapers: Domain-specific crawlers (news, e-commerce, social media)
  - utils: Logging, validation, and reporting utilities

Example:
    >>> from nepali_scraper.scrapers import RatopatiScraper
    >>> from nepali_scraper.config import PLATFORM_CONFIG
    >>> scraper = RatopatiScraper(PLATFORM_CONFIG["ratopati"])
    >>> scraper.scrape_articles(max_pages=5)
    >>> scraper.save_checkpoint(Path("data/raw_corpus.jsonl"))

Version: 1.0.0
Author: Elite NLP Research Team
License: Academic Use
"""

__version__ = "1.0.0"
__author__ = "Elite NLP Research Team"

from . import config
from . import processors
from . import scrapers
from . import utils

__all__ = [
    "config",
    "processors",
    "scrapers",
    "utils",
]
