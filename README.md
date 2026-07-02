# Nepali Sentiment Analysis - Web Scraping Architecture

## Overview

This repository contains **Phase 1** of a comprehensive Nepali sentiment analysis project: a production-grade web scraping system designed to collect 12,000–15,000 raw sentences from three distinct domains, which will yield 4,000–5,000 clean, human-verified sentences after processing.

**Project Goal**: Build and benchmark a four-tier model ladder (TF-IDF, fastText, mBERT, MuRIL) to publish a rigorous academic paper on low-resource Nepali NLP with cross-domain transfer analysis.

---

## Architecture Overview

### Three-Domain Collection Strategy

The system implements balanced crawling across three target sectors:

| Domain | Platforms | Target Raw | Expected Clean | Target % |
|--------|-----------|-----------|----------------|----------|
| **News & Politics** | Ratopati, Setopati, Onlinekhabar | 5,000 | 2,000–2,500 | 33.3% |
| **E-commerce Reviews** | Daraz | 5,000 | 2,000–2,500 | 33.3% |
| **Social Media** | Twitter/X (lang:ne) | 4,000 | 1,600–2,000 | 33.3% |
| **TOTAL** | — | 12,000–15,000 | **4,000–5,000** | 100% |

### Data Loss Model

Standard web scraping in low-resource environments suffers **50–60% data loss** from:
- Encoding/Unicode normalization issues
- HTML parsing artifacts
- Script-mixing (Devanagari + Latin romanization)
- Duplicate records
- Out-of-length sentences

Thus, collection targets are set **2.5–3× higher** to achieve final dataset size.

---

## Project Structure

```
d:\ds mdoel\
├── nepali_scraper/                    # Main Python package
│   ├── __init__.py
│   ├── config.py                      # Centralized configuration
│   ├── processors.py                  # NepaliTextProcessor (NFC normalization, filtering)
│   ├── scrapers.py                    # Domain-specific scrapers
│   ├── utils.py                       # Logging, validation, reporting utilities
│   └── README.md                      # Package documentation
├── data/                              # Output directory (created automatically)
│   ├── 01_raw_corpus.jsonl            # Raw scraped text
│   ├── 02_cleaned_corpus.jsonl        # After text processing pipeline
│   ├── 03_validation_partition.jsonl  # 20% for dual-annotator agreement
│   ├── 04_annotated_dataset.jsonl     # After human annotation + kappa validation
│   ├── 05_train_split.jsonl           # 80% training
│   └── 06_test_split.jsonl            # 20% held-out test
├── logs/                              # Log files (created automatically)
│   └── scraper.log
├── scraper_main.py                    # Main orchestration script
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## Core Components

### 1. **config.py** – Centralized Configuration

```python
from nepali_scraper.config import DOMAIN_TARGETS, PLATFORM_CONFIG

# Access domain collection targets
print(DOMAIN_TARGETS["news_politics"]["raw_target"])  # 5000

# Access platform-specific settings
print(PLATFORM_CONFIG["ratopati"]["max_pages"])  # 15
```

**Key Features**:
- Platform-specific URLs and CSS selectors
- Rate-limiting and retry logic parameters
- Text processing thresholds (min/max length)
- Annotation parameters (Cohen's Kappa targets)

### 2. **processors.py** – NepaliTextProcessor

Implements the production text processing pipeline:

```python
from nepali_scraper.processors import NepaliTextProcessor

processor = NepaliTextProcessor({
    "min_length": 15,
    "max_length": 350
})

# NFC normalization + Devanagari filtering
cleaned = processor.clean_and_normalize("मेरो नाम  राज छ !!! 😊")
# Output: "मेरो नाम राज छ !"

# Sentence segmentation by Nepali punctuation (।, ?, !)
sentences = processor.segment_sentences(paragraph)
# Output: List of cleaned sentences
```

**Algorithm**:
1. **Unicode NFC Normalization** – Ensures canonical form consistency
2. **Script Filtering** – Removes non-Devanagari noise while preserving ASCII (numbers)
3. **Whitespace Collapse** – Normalizes spacing
4. **Sentence Tokenization** – Splits by (।, ?, !)
5. **Length Filtering** – Keeps sentences 15–350 characters

### 3. **scrapers.py** – Domain-Specific Crawlers

#### News & Politics Scrapers

```python
from nepali_scraper.scrapers import RatopatiScraper

scraper = RatopatiScraper(config=PLATFORM_CONFIG["ratopati"])
scraper.scrape_articles(max_pages=15)
scraper.save_checkpoint(Path("data/01_raw_corpus.jsonl"))
```

**Platforms**:
- **Ratopati** (`ratopati.com`)
- **Setopati** (`setopati.com`)
- **Onlinekhabar** (`onlinekhabar.com`)

**Strategy**: Paginated article listing → fetch full content → extract body text

#### E-commerce Scraper

```python
from nepali_scraper.scrapers import DarazScraper

scraper = DarazScraper(config=PLATFORM_CONFIG["daraz"])
scraper.scrape_product_reviews(max_products=500)
```

**Platform**: Daraz Nepal (`daraz.com.np`)

**Strategy**: Browse categories → iterate products → extract customer reviews

#### Social Media Scraper

```python
from nepali_scraper.scrapers import TwitterScraper

scraper = TwitterScraper(
    config=PLATFORM_CONFIG["twitter"],
    bearer_token="YOUR_API_TOKEN"
)
scraper.scrape_nepali_tweets(query="lang:ne", max_results=5000)
```

**Platform**: Twitter/X

**Strategy**: Use Twitter API v2 with `lang:ne` filter for Nepali tweets

**Requires**: Twitter API v2 Bearer token ([Get from developer.twitter.com](https://developer.twitter.com/en/portal/dashboard))

### 4. **utils.py** – Utilities

```python
from nepali_scraper.utils import (
    setup_logging,
    validate_jsonl_file,
    generate_domain_balance_report,
    print_report
)

# Logging setup
logger = setup_logging(Path("logs/scraper.log"), level="INFO")

# Validate JSONL integrity
stats = validate_jsonl_file(Path("data/01_raw_corpus.jsonl"))
print(f"Valid: {stats['valid_lines']}, Invalid: {stats['invalid_lines']}")

# Generate domain distribution report
report = generate_domain_balance_report(Path("data/01_raw_corpus.jsonl"))
print_report(report, "Domain Balance Report")
```

**Functions**:
- `setup_logging()` – Configure logging to file + console
- `validate_jsonl_file()` – Verify JSONL integrity
- `deduplicate_jsonl()` – Remove duplicates
- `generate_domain_balance_report()` – Analyze distribution
- `verify_dependencies()` – Check Python package availability

### 5. **scraper_main.py** – Orchestrator

Master script that coordinates the entire pipeline:

```bash
# Full pipeline
python scraper_main.py

# Sample mode (10 items per domain for testing)
python scraper_main.py --sample

# With Twitter API token
python scraper_main.py --twitter-token=YOUR_TOKEN

# Debug logging
python scraper_main.py --debug
```

**Workflow**:
1. Check environment and dependencies
2. Load API credentials from `.env` or arguments
3. Execute domain-specific scrapers in sequence
4. Apply text processing pipeline to raw corpus
5. Generate comprehensive reports

---

## Installation & Setup

### Step 1: Install Dependencies

```bash
cd d:\ds mdoel
pip install -r requirements.txt
```

**Required Packages**:
- `requests` – HTTP client
- `beautifulsoup4` – HTML parsing
- `numpy`, `pandas` – Data structures
- `scikit-learn` – ML utilities
- `torch`, `transformers` – For later model stages

### Step 2: (Optional) Set Up Twitter API Credentials

Create `.env` file in project root:

```env
TWITTER_BEARER_TOKEN=YOUR_BEARER_TOKEN_HERE
```

Or pass via CLI:

```bash
python scraper_main.py --twitter-token=YOUR_TOKEN
```

### Step 3: Run Scraper

```bash
# Test mode (small scale)
python scraper_main.py --sample

# Full run
python scraper_main.py
```

---

## Data Format

All outputs are **JSONL** (JSON Lines — one JSON object per line):

### Raw Corpus Format

```jsonl
{"domain": "news_politics", "platform": "ratopati", "url": "https://...", "headline": "...", "content": "...", "scraped_at": "2026-06-20T12:00:00"}
{"domain": "ecommerce_reviews", "platform": "daraz", "product_name": "...", "review_text": "...", "url": "https://...", "scraped_at": "2026-06-20T12:01:00"}
{"domain": "social_media", "platform": "twitter", "tweet_id": "123", "text": "...", "created_at": "2026-06-20T12:02:00", "scraped_at": "2026-06-20T12:03:00"}
```

### Cleaned Corpus Format

```jsonl
{"domain": "news_politics", "platform": "ratopati", "sentence": "नेपालको अर्थनीति बढदारी गरिरहेको छ।", "source_url": "https://...", "processed_at": "2026-06-20T12:30:00"}
{"domain": "ecommerce_reviews", "platform": "daraz", "sentence": "यो उत्पाद धेरै राम्रो छ र मलाई मनपर्यो।", "source_url": "https://...", "processed_at": "2026-06-20T12:31:00"}
```

---

## Processing Pipeline Details

### Text Normalization Algorithm

```
Input: "मेरो नाम  राज छ !!! 😊 babal"
  ↓
NFC Normalization
  ↓
Script Filtering (keep [॥०-॥xFF || ॥०००-॥०९७F] only)
  ↓
Devanagari Collapsing: "मेरो नाम  राज छ !!!"
  ↓
Whitespace Collapsed: "मेरो नाम राज छ !"
Output: "मेरो नाम राज छ !"
```

### Sentence Segmentation

```
Input: "मलाई सेब पनि राम्रो लाग्यो। आलु पनि राम्रो छ। तपाई कस्तो छौ?"
  ↓
Split by [।, ?, !]
  ↓
Reconstruct sentences
  ↓
Length filter (15 < len < 350)
  ↓
Output:
  ["मलाई सेब पनि राम्रो लाग्यो।",
   "आलु पनि राम्रो छ।",
   "तपाई कस्तो छौ?"]
```

### Data Loss Accounting

**Expected loss per stage**:
- HTML parsing artifacts: ~10%
- Non-Nepali text filtering: ~15%
- Duplicate records: ~10%
- Length filtering: ~15%
- **Total**: ~50%

**Collection targets compensate** by targeting 2.5–3× final size.

---

## Quality Assurance

### Validation Metrics

After scraping, verify:

```bash
# Check JSONL integrity
python -c "
from nepali_scraper.utils import validate_jsonl_file
from pathlib import Path
stats = validate_jsonl_file(Path('data/01_raw_corpus.jsonl'))
print(f'Valid: {stats[\"valid_lines\"]}, Invalid: {stats[\"invalid_lines\"]}')
"

# Check domain distribution
python -c "
from nepali_scraper.utils import generate_domain_balance_report, print_report
from pathlib import Path
report = generate_domain_balance_report(Path('data/01_raw_corpus.jsonl'))
print_report(report, 'Domain Balance')
"
```

### Expected Results (Full Run)

```
Raw Corpus:
  - Total records: 12,000–15,000
  - news_politics: ~5,000 (33%)
  - ecommerce_reviews: ~5,000 (33%)
  - social_media: ~4,000 (34%)

Cleaned Corpus:
  - Total sentences: ~4,000–5,000 (after 50% loss)
  - Avg length: ~80–120 characters
  - All valid Devanagari script
```

---

## Next Phases (Roadmap)

This Phase 1 sets up for subsequent stages:

- **Phase 2**: Human-in-the-loop annotation with inter-annotator agreement (Cohen's Kappa)
- **Phase 3**: Four-tier model benchmarking (TF-IDF → fastText → mBERT → MuRIL)
- **Phase 4**: Cross-domain transfer analysis & error classification
- **Phase 5**: Publication-grade manuscript (5-page ACM/IEEE format)

---

## Troubleshooting

### Missing Packages

```bash
pip install -r requirements.txt
```

### Network Issues / Rate Limiting

Scraper implements exponential backoff. If still failing:
- Increase `REQUEST_DELAY_BASE` in `config.py`
- Reduce `max_pages` / `max_products` arguments
- Run in `--sample` mode first

### Twitter API Errors

```
⚠ Twitter API Bearer token not configured
```

**Solution**: Set `TWITTER_BEARER_TOKEN` environment variable or pass `--twitter-token` CLI arg.

### Out of Memory with Large Datasets

Process data in chunks using the `batch_mode`:

```python
processor = NepaliTextProcessor()
batch_size = 1000
# Process batches separately
```

---

## References & Resources

### Nepali NLP Benchmarks
- NepaliBERT benchmark suite
- Ai4Bharat language models
- Facebook fastText (Nepali `cc.ne.300.bin`)

### Key Papers (for §2. RELATED WORK)
- Cross-lingual transfer in low-resource NLP (EMNLP, ACL)
- Sentiment analysis in morphologically rich languages
- Character-level models for Devanagari script

### Unicode Standards
- [Devanagari Unicode Block (U+0900–U+097F)](https://unicode.org/charts/PDF/U0900.pdf)
- Python `unicodedata` module: [NFC normalization](https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize)

---

## Contributing

To extend with new platforms:

1. Create new scraper class inheriting `BaseScraper`
2. Implement `scrape_*()` method
3. Register in `config.py` → `PLATFORM_CONFIG`
4. Add orchestration logic in `scraper_main.py`

---

## License

Academic use only. Part of publishable sentiment analysis research.

---

## Author

Elite AI Research Scientist & Senior NLP Engineer  
Specialized in South Asian language NLP  
Nepali Sentiment Analysis Project — Phase 1 (Web Scraping)

**Generated**: June 20, 2026

---

## Appendix A: Expected Project Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Phase 1** (Current) | 1–2 weeks | Raw 12K–15K corpus |
| **Phase 2** | 2–3 weeks | Annotated dataset (4K–5K verified) |
| **Phase 3** | 3–4 weeks | Model benchmarks + confusion matrices |
| **Phase 4** | 1–2 weeks | Cross-domain transfer + error analysis |
| **Phase 5** | 1–2 weeks | Publication-ready manuscript |

**Total**: 8–13 weeks to publishable paper.

---

## Appendix B: Sample Test Run

```bash
# Verify setup
python -c "from nepali_scraper import config; print(f'Project root: {config.PROJECT_ROOT}')"

# Run sample (10 items per domain)
python scraper_main.py --sample

# Expected output (sample mode):
# [INFO] NEPALI SENTIMENT ANALYSIS - WEB SCRAPING ORCHESTRATION
# [INFO] Checking environment and dependencies...
# [INFO] ✓ requests
# [INFO] ✓ bs4
# ...
# [INFO] Scraping Ratopati (max 5 pages)...
# [INFO] ✓ Ratopati: 15 articles, 0 errors
# [INFO] TEXT PROCESSING PIPELINE
# [INFO] ✓ Processed 45 raw records → 287 cleaned sentences
# ...
```

---

*End of README*
