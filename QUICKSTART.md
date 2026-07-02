# Quick Start Guide - Nepali Scraper

## 5-Minute Setup

### 1. Install Dependencies (2 min)

```bash
cd d:\ds mdoel
pip install -r requirements.txt
```

### 2. Verify Installation (1 min)

```bash
python -c "from nepali_scraper import config; print(f'✓ Setup verified. Project root: {config.PROJECT_ROOT}')"
```

### 3. Run Sample Test (1 min)

```bash
python scraper_main.py --sample
```

This runs the full pipeline with only 10 items per domain to verify everything works.

### 4. Run Full Collection (Varies)

```bash
python scraper_main.py
```

**Expected collection**:
- News & Politics: ~5,000 articles
- E-commerce Reviews: ~5,000 reviews
- Social Media: ~4,000 tweets (if Twitter API enabled)
- **Total**: ~12,000–15,000 raw records → **~4,000–5,000 clean sentences**

---

## Optional: Enable Twitter Scraping

Create `.env` file in project root:

```
TWITTER_BEARER_TOKEN=YOUR_API_TOKEN_HERE
```

Get your token from: https://developer.twitter.com/en/portal/dashboard

---

## Outputs

After running, check:

```bash
# View collected data
ls -la d:\ds mdoel\data\

# Check raw corpus stats
python -c "
from nepali_scraper.utils import generate_domain_balance_report, print_report
from pathlib import Path
report = generate_domain_balance_report(Path('data/01_raw_corpus.jsonl'))
print_report(report, 'Collection Summary')
"

# View logs
tail -50 d:\ds mdoel\logs\scraper.log
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'requests'` | Run `pip install -r requirements.txt` |
| `⚠ Twitter API Bearer Token not configured` | Twitter scraping is optional; continue. To enable, add token to `.env` |
| Network timeout errors | Increase rate limit or use `--sample` mode |
| Out of memory | Reduce `max_pages` in `config.py` |

---

## Next: Phase 2 (Human Annotation)

After data collection completes:

```python
# Phase 2 will include:
# - Dual-annotator workflow
# - Cohen's Kappa agreement calculation
# - Data validation and consensus mediation
```

Reference: See `Phase 2: Human-In-The-Loop Annotation & Mathematical Validation` in the main blueprint.

---

*For detailed documentation, see README.md*
