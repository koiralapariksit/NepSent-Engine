# NepSent-Engine Architecture Specification (v2.0)

**Document Version:** 2.0.0  
**Status:** Implemented & Verified (64/64 Unit Tests Passing)  
**Schema Standard:** `2.0-ordinal` (`0: Negative < 1: Neutral < 2: Positive`)  

---

## 1. System Overview

NepSent-Engine transforms low-resource Nepali sentiment analysis from ad-hoc scripts into a mathematically sound, reproducible research platform.

```
                           +------------------------+
                           |  Raw Scraped Content   |
                           +-----------+------------+
                                       |
                                       v
               +------------------------------------------------+
               |  nepali_scraper/collectors.py                  |
               |  - DatasetIngestor                             |
               |  - Provenance (source, license, timestamp)     |
               |  - Exact (SHA-256) & Near-Dup (Jaccard >= 0.85)|
               |  - Hindi Function Word Filter                  |
               +-----------------------+------------------------+
                                       |
                                       v
               +------------------------------------------------+
               |  nepali_scraper/processors.py                  |
               |  - Unicode NFC Normalization                   |
               |  - ZWJ (\u200D) & ZWNJ (\u200C) Preservation   |
               |  - Inclusive Length Filtering [15, 350]        |
               |  - Script Ratio Enforcement (>= 0.70)          |
               +-----------------------+------------------------+
                                       |
                                       v
               +------------------------------------------------+
               |  nepali_scraper/annotation.py                  |
               |  - Double-Blind Sheet Exporter                 |
               |  - Cohen's Kappa (Unweighted, Linear, Quad)    |
               |  - Disagreement Adjudication Engine            |
               +-----------------------+------------------------+
                                       |
                                       v
               +------------------------------------------------+
               |  nepali_scraper/splitter.py                    |
               |  - Hash-Grouped Stratified Splitter            |
               |  - Zero Train-Test Sentence Leakage            |
               |  - Minimum Per-Class Test Support Guard (>= 80)|
               +-----------------------+------------------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
+------------------------------------+  +------------------------------------+
| nepali_scraper/models/classical.py |  | nepali_scraper/models/transformers |
| Level 0: Majority Baseline         |  | Level 3: Multilingual mBERT        |
| Level 1: TF-IDF LogReg & LinSVC    |  | Level 3: Indic MuRIL               |
| Level 2: Subword FastText          |  | Level 4: Monolingual NepBERTa      |
+------------------+-----------------+  +------------------+-----------------+
                   |                                       |
                   +-------------------+-------------------+
                                       |
                                       v
               +------------------------------------------------+
               | nepali_scraper/evaluation/                     |
               | - metrics.py: B=2,000 Percentile Bootstrap CIs |
               | - metrics.py: Paired Bootstrap & McNemar Tests |
               | - cross_domain.py: Leave-One-Domain-Out (LODO) |
               | - linguistic_analysis.py: Morphological Neg    |
               +-----------------------+------------------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
+------------------------------------+  +------------------------------------+
| nepali_scraper/cli.py (`nepsent`)  |  | app.py (Honest Streamlit Demo)     |
| Unified CLI Pipeline Automation    |  | Live Telemetry & Calibration Notice|
+------------------------------------+  +------------------------------------+
```

---

## 2. Module Specifications

### 2.1 Configuration (`nepali_scraper/config.py`)
- **Dynamic Project Resolution**: All paths derived dynamically using `Path(__file__).resolve().parents[1]`. Zero hardcoded drive letters or absolute paths.
- **Ordinal Schema Definition**:
  - `0: Negative`
  - `1: Neutral`
  - `2: Positive`
  - Strict preservation of ordinal distances: distance $|0 - 2| = 2$, while $|0 - 1| = 1$.
- **Legacy Backward Compatibility**: Preserves `LEGACY_LABEL_SCHEMA` (`1.0-nominal`) for historical audit trace reading without corrupting active pipelines.

### 2.2 Text Processing (`nepali_scraper/processors.py`)
- **Unicode NFC Normalization**: Decompositions normalized to canonical composite codepoints.
- **Zero-Width Character Preservation**:
  - Strips non-semantic invisible whitespace (BOM `\uFEFF`, ZWSP `\u200B`, LRM `\u200E`, RLM `\u200F`).
  - Preserves Zero-Width Joiner (ZWJ `\u200D`) and Zero-Width Non-Joiner (ZWNJ `\u200C`) required for Devanagari conjuncts and eyelash-ra (e.g. `र्या`, `र्‍य`).
- **Boundary Preservation**: Sentence segmentation preserving Purna Viram (`।`), Deergha Viram (`॥`), question marks, and exclamations without dropping trailing sentences.
- **Deterministic Sentence Hashing**: Canonical NFC text hashed via SHA-256 for leakage-free group partitioning.

### 2.3 Data Ingestion & Provenance (`nepali_scraper/collectors.py`)
- **DatasetIngestor**:
  - Validates provenance: `source`, `collection_timestamp`, `license`, and `domain`.
  - Canonical domain taxonomy: `news_politics`, `ecommerce_reviews`, `social_media`.
  - Exact deduplication via SHA-256 hash sets.
  - Near-duplicate detection via character 3-gram Jaccard similarity ($\ge 0.85$).
  - Hindi contamination detector utilizing high-frequency Hindi-exclusive function words (`है`, `हैं`, `था`, `थी`, `क्योंकि`, `लेकिन`, `नहीं`, `किया`).

### 2.4 Annotation & Agreement (`nepali_scraper/annotation.py`)
- **Double-Blind Sheet Exporter**: Shuffles sentences, generates deterministic record IDs (`nep_<hash[:12]>`), and blinds true source labels to prevent annotator bias.
- **Multi-Metric IAA**:
  - Unweighted Cohen's $\kappa$
  - Linear Weighted $\kappa_w$ (penalizes $0 \leftrightarrow 2$ twice as heavily as $0 \leftrightarrow 1$)
  - Quadratic Weighted $\kappa_w$
  - Per-domain agreement breakdowns to pinpoint genre-specific ambiguities.
- **Disagreement Adjudication Engine**: Flags discordant pairs and applies third-party tie-breaking rules.

### 2.5 Leakage-Free Stratified Splitting (`nepali_scraper/splitter.py`)
- **Hash-Grouped Splitting**: All occurrences of identical canonical sentence hashes are forced into the same partition, eliminating train/validation/test contamination.
- **Statistical Power Alert**: Verifies Cochran's minimum test support. Slices with $< 80$ samples per class emit high-variance warnings.

### 2.6 Model Benchmark Ladder (`nepali_scraper/models/`)
- **Level 0 (Empirical Lower Bound)**: `MajorityBaselineClassifier` — estimates lower bound against class imbalance.
- **Level 1 (N-gram Linear Models)**: `TfidfLinearModel` — word n-grams `(1, 2)` + character n-grams `(2, 4)`.
  - L2-regularized `LogisticRegression` with balanced class weights.
  - Platt-calibrated `LinearSVC` with dynamic fold adaptation (`cv = min(self.cv, min_class_count)`).
- **Level 2 (Subword Morphological Model)**: `SubwordFastTextClassifier` — character n-grams `(3, 5)` capturing inflectional suffixes without C++ compiler dependencies.
- **Level 3 & 4 (Pretrained Deep Transformers)**: `TransformerSentimentTrainer` — unified training harness for `bert-base-multilingual-cased` (mBERT), `google/muril-base-cased` (MuRIL), and `Rajan/NepBERTa`.

### 2.7 Statistical Evaluation & Linguistics (`nepali_scraper/evaluation/`)
- **Non-Parametric Bootstrap Engine**: $B = 2,000$ empirical percentile resamples for Macro-F1, Accuracy, and per-class F1.
- **Hypothesis Testing**: Paired difference bootstrap test and McNemar's contingency test (exact binomial & continuity-corrected $\chi^2$).
- **Leave-One-Domain-Out (LODO)**: Cross-domain generalization evaluator across distinct genres.
- **Nepali Morphological Negation Detector**: Word-boundary matching of standalone auxiliaries (छैन, होइन, थिएन, हुँदैन), inflectional verbal suffixes (`-[े]न`, `-एन`, `-न्न`, `-दिन`), and prefixed verbs (नगर्नुहोला, नकिन्नुहोला) without false-firing on nouns with 'न' (नेपाल, नयाँ, नदी, नागरिक).
- **Subword Fertility Telemetry**: Quantifies tokens per word, char-to-token ratios, and `[UNK]` frequencies.
- **Linguistic Error Taxonomy**: Categorizes errors into 6 diagnostic buckets.

### 2.8 Interfaces & Applications
- **CLI (`nepali_scraper/cli.py`)**: `nepsent {ingest, split, iaa, train, evaluate, predict}`.
- **Streamlit App (`app.py`)**: Honest demonstration application with anti-fabrication demo mode blocks and probability calibration transparency.
