# Contributing to NepSent-Engine

Thank you for your interest in contributing to **NepSent-Engine**! We welcome contributions from researchers, software engineers, and native Nepali linguists.

Our core mission is to establish **scientifically credible, reproducible, and leakage-free Nepali NLP tools**. To protect research validity, all contributions must adhere to our engineering and research standards.

---

## 1. Code of Conduct

All contributors and participants agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please maintain a welcoming, respectful, and constructive environment.

---

## 2. Fundamental Scientific Invariants

Any pull request modifying data processing, annotation, model architectures, or evaluation MUST satisfy these invariants:

1. **Zero Data Leakage:**
   - Datasets must be split using sentence-hash grouping (`nepali_scraper/splitter.py`). Exact duplicate sentences or paraphrased near-duplicates must never cross train/validation/test partitions.
2. **Schema Invariance (`2.0-ordinal`):**
   - All active code must strictly use the ordinal mapping: `0: Negative, 1: Neutral, 2: Positive`. Nominal mappings (`0: Negative, 1: Positive, 2: Neutral`) break weighted Cohen's Kappa ($\kappa_w$) and ordinal distances.
3. **True Non-Parametric Bootstrap Evaluation:**
   - Benchmark reports must report empirical percentile bootstrap confidence intervals ($B = 2,000$ resamples) for Macro F1. Normal approximation assumptions on Macro F1 are mathematically invalid.
4. **Devanagari Unicode Rigor:**
   - Text processing must preserve essential zero-width conjunct joiners (ZWJ `\u200D` and ZWNJ `\u200C`) while removing non-semantic invisible spaces.
5. **No Synthetic Benchmarks:**
   - PRs must never commit fake benchmark numbers, simulated annotator outputs, or hardcoded metrics.

---

## 3. Development Workflow

### Setting up the Environment

```bash
# 1. Clone repository
git clone https://github.com/koiralapariksit/NepSent-Engine.git
cd NepSent-Engine

# 2. Create virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install pytest streamlit
```

### Running Tests

Before submitting any code changes, ensure all tests pass:

```bash
pytest -v
```

All 64+ existing tests must pass with a 100% pass rate. If you add a new feature or algorithm, you must include unit tests in `tests/`.

---

## 4. Submitting Pull Requests

1. Fork the repository and create your branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Write clean, modular Python with standard docstrings and type annotations.
3. Never introduce import-time network calls or unbuffered stdout printing.
4. Run the full test suite (`pytest -v`).
5. Open a Pull Request on GitHub with a clear description of:
   - What problem your PR solves.
   - Evidence that tests pass.
   - Any impact on reproducibility or schema.

