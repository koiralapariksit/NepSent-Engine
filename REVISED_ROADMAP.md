# NepSent-Engine: Revised Research-First Implementation Roadmap

**Document:** `REVISED_ROADMAP.md`  
**Status:** Architecture Blueprint & Implementation Schedule  
**Target Completion:** Academic Research Benchmark & Reproducible Open-Source Engine  
**Guiding Principle:** *Research integrity > reproducibility > data quality > model performance > UI polish*

---

## 1. Executive Summary & Philosophy Shift

Following our Phase Zero repository audit and the subsequent independent critical review, NepSent-Engine transitions from an **engineering-first demonstration scaffold** to a **scientifically grounded NLP research project**.

### Key Changes from the Original Plan:
1. **Replaced Waterfall Ordering:** Testing, validation, and ethical safeguards are integrated into every phase rather than deferred to an end-stage phase.
2. **De-risked via Dual Pilots:** Implemented a Pilot Collection (50–100 genuine records) and a Pilot Annotation (30–50 double-annotated sentences) prior to full-scale commitments.
3. **Prevented Evaluation Leakage:** Mandatory sentence-level SHA256 grouping across all data partitions.
4. **Methodological Rigor:** Swapped ad-hoc performance targets for 95% bootstrap confidence intervals, paired statistical tests, ordinal weighted Kappa ($\kappa_w$), and true Leave-One-Domain-Out (LODO) cross-domain generalization.
5. **Preserved Ambiguity:** Discontinued the flawed practice of discarding disagreed annotations; ambiguous and consensus partitions are evaluated distinctly to audit model robustness.

---

## 2. Scientific Phase Structure & Milestones

```
Phase 0 ──► Phase 0.5 ──► Phase 1.0 ──► Phase 1.5 ──► Phase 2.0 ──► Phase 2.5 ──► Phase 3.0 ──► Phase 3.5 ──► Phase 4.0 ──► Phase 5.0
 Audit       Repo Hygiene  Data Eng &   Pilot Data    Pilot Human   Full Corpus   Leak-Free     Linguistic   Inference    Ethics, Data
 (DONE)      & Protocol    Unit Tests   Collection    Annotation    Annotation    Benchmarking  Error Audit  Demo App     Statement, OSS
```

---

### Phase 0: Baseline Audit & Critical Review (COMPLETED)
- **Delivered Artifacts:**
  - `AUDIT_REPORT.md` (Identified 7 critical architectural/data bugs).
  - `CRITICAL_REVIEW.md` (Identified 15 additional gaps, statistical flaws, and methodology oversights).
  - `LITERATURE_REVIEW_PLAN.md` (Survey of verified literature: NEPSEE, NepBERTa, NepEMO, etc.).
  - `RESEARCH_GAP.md` (Formal contribution thesis and scope boundaries).
  - `RISK_REGISTER.md` (Comprehensive operational and technical risk matrix).

---

### Phase 0.5: Repository Hygiene, Licensing & Protocol Pre-Registration
*Goal: Secure the repository environment, eliminate credential risk, and preserve historical artifacts before writing code.*

- **Action Items:**
  1. **Add `.gitignore`:**
     - Exclude `.env`, `venv/`, `__pycache__/`, `*.pt`, `*.bin`, `*.safetensors`, `data/scratch/`.
  2. **Add `LICENSE`:**
     - Adopt Apache 2.0 (or MIT) for codebase; CC BY-SA 4.0 for curated dataset records.
  3. **Preserve Legacy Test Artifacts:**
     - Safely relocate current synthetic data (`01_raw_corpus.jsonl` through `09_model_comparison.json`) into `data/legacy_simulated/` with metadata marking them as `SYNTHETIC_TEST` and `SIMULATED_ANNOTATION`. Never delete or overwrite them.
  4. **Eliminate Hardcoded Paths:**
     - Replace all `d:/ds mdoel/data/` references with dynamic `PROJECT_ROOT` imports from `nepali_scraper.config`.
  5. **Mute Import Pollution:**
     - Remove stdout print statements executed on import in `nepali_scraper/config.py`.
  6. **Experiment Tracking Foundation:**
     - Create lightweight experiment logging schema (`experiments/runs/<timestamp>_<model>.json`) capturing git commit hash, random seed, hyperparameter dictionary, and hardware specs.
- **Verification Gate:**
  - `git status` verifies `.env` and `venv/` are ignored.
  - Test import of `nepali_scraper.config` produces clean stdout.

---

### Phase 1.0: Data Engineering with Integrated Unit Tests
*Goal: Build robust, ethical Devanagari text processing and data collection modules backed by rigorous automated tests.*

- **Action Items:**
  1. **Fix `NepaliTextProcessor` (`nepali_scraper/processors.py`):**
     - Fix off-by-one boundary bug: change `min_length < len < max_length` to `min_length <= len <= max_length`.
     - Retain sentences lacking trailing delimiters (`।`, `?`, `!`) without silent dropping.
     - Enforce Unicode NFC normalization while **explicitly preserving the virama (`्`) and non-spacing diacritics** (addressing tokenizer stripping identified by Baral et al., 2026).
     - Preserve Romanized and English code-mixed tokens without destructive regex mangling.
  2. **Implement Group Hashing & Deduplication:**
     - Compute deterministic `sentence_hash = sha256(normalized_text.encode('utf-8')).hexdigest()`.
     - Group identical sentences across all source variants.
  3. **Build Integrated Test Suite (`tests/`):**
     - `tests/test_processors.py`: Test sentence segmentation, boundary lengths, NFC normalization, punctuation handling.
     - `tests/test_dedup.py`: Test hash collision handling, whitespace invariance, and group assignment.
  4. **Feasibility-Checked Collection Engine:**
     - Implement clean, ethical collection adapters respecting `robots.txt` and rate limits:
       - **News & Politics:** Official RSS feeds and public sitemaps (Ratopati, Setopati, Onlinekhabar).
       - **E-Commerce Reviews:** Static public review extractors or verified open Indic review corpora.
       - **Social Media:** Public academic dumps (NepCOV19Tweets, Reddit public dumps) avoiding paid Twitter API gates.
- **Verification Gate:**
  - `pytest tests/` passes 100% on synthetic and real test strings before running collection.

---

### Phase 1.5: Pilot Data Collection & Dataset Size Justification
*Goal: Collect and validate a miniature real-world corpus and establish statistical sample size power calculations.*

- **Action Items:**
  1. **Execute Pilot Collection:**
     - Collect exactly **50–100 genuine records** (target: ~25 news, ~25 e-commerce, ~25 social media).
     - Assign provenance metadata: `REAL_COLLECTED`, source URL/identifier, collection timestamp, language tag (`ne`).
  2. **Data Quality Validation (`DATA_QUALITY_REPORT.md`):**
     - Measure character distributions, word count percentiles, and vocabulary overlap across the 3 domains.
  3. **Formal Statistical Power & Dataset Size Justification:**
     - *Mathematical Justification:* For a 3-class classification problem ($\alpha = 0.05$):
       - To achieve a test set confidence margin $E \le \pm 5\%$ on Macro-F1 at 95% confidence, binomial theory dictates:
         $$N_{\text{test}} \ge \frac{z^2 \cdot p(1-p)}{E^2} = \frac{(1.96)^2 \cdot (0.5)(0.5)}{(0.05)^2} \approx 384 \text{ samples}$$
       - Allocating an 80/20 train/test split: Total dataset target $N \approx 1,920 \approx 2,000$ sentences (~650 per domain).
       - *Short-term achievable milestone:* 1,000 verified sentences ($N_{\text{test}} = 200$, CI margin $\approx \pm 6.9\%$).
       - *Long-term target:* 3,000 verified sentences ($N_{\text{test}} = 600$, CI margin $\approx \pm 4.0\%$).
- **Verification Gate:**
  - All pilot sentences verified as genuine Nepali by native speakers, zero empty or duplicate records.

---

### Phase 2.0: Pilot Annotation Study (2 Independent Human Annotators)
*Goal: Calibrate human annotation guidelines, compute pilot inter-annotator agreement, and resolve ambiguities.*

- **Action Items:**
  1. **Formalize Annotation Guidelines (`docs/ANNOTATION_GUIDELINES.md`):**
     - Detailed 3-class ordinal schema: `0: Negative`, `1: Neutral`, `2: Positive`.
     - Explicit decision rules for:
       - Factual news reporting vs. editorial opinion.
       - Sarcasm and irony (mark intent over literal words; tag `is_sarcastic: true`).
       - Mixed sentiment compound sentences (terminal clause priority rule).
       - Code-mixed English/Nepali expressions (e.g., "product ekdam ramro cha but delivery late vayo").
  2. **Double-Blind Pilot Annotation:**
     - Select 30–50 sentences from the pilot corpus spanning all three domains.
     - Two independent human annotators label the sentences in total isolation.
  3. **Agreement Metric Computation:**
     - Compute Unweighted Cohen's $\kappa$ and Linear-Weighted $\kappa_w$.
     - Compute Krippendorff's $\alpha$.
     - Generate pilot confusion matrix to identify systematic confusion pairs (e.g., Neutral vs. Negative).
  4. **Adjudication & Calibration Meeting:**
     - Annotators and researcher discuss all disagreed sentences.
     - Document specific points of confusion and refine guidelines with concrete examples.
- **Verification Gate:**
  - Pilot achieves substantial ordinal agreement ($\kappa_w \ge 0.60$ or documented calibration) before scaling full annotation.

---

### Phase 2.5: Full Human Annotation Campaign & Ambiguity Partitioning
*Goal: Scale annotation to the target corpus, implement formal adjudication, and avoid consensus selection bias.*

- **Action Items:**
  1. **Streamlit Annotation Tool (`annotation_app/`):**
     - Lightweight local Streamlit app supporting:
       - Blinded annotator login.
       - Random presentation with session progress saving.
       - Confidence rating (Low / Medium / High) and tags (`sarcasm`, `code_mixed`, `negation`).
  2. **Double-Annotated Core Partition:**
     - A minimum of 20–30% of the corpus is labeled by both annotators to measure global and per-domain agreement.
     - Remainder split between annotators for scaling.
  3. **Preserve Ambiguity (No Selective Discarding):**
     - Unlike NEPSEE (which discarded 43% of disagreed data), NepSent-Engine preserves all data into two structured partitions:
       - **Partition A (Consensus Set):** Sentences where both annotators agreed without adjudication.
       - **Partition B (Adjudicated Set):** Sentences where annotators disagreed, resolved by a third senior adjudicator or joint consensus, retaining the `original_disagreement` metadata.
  4. **Agreement Reporting:**
     - Report global $\kappa$, global $\kappa_w$, and per-domain $\kappa_w$ for News, E-Commerce, and Social Media separately in `ANNOTATION_REPORT.md`.
- **Verification Gate:**
  - Published annotation report with complete confusion matrices, marginal distributions, and adjudication logs.

---

### Phase 3.0: Leakage-Free Model Ladder & Benchmarking
*Goal: Implement a rigorous 4-tier model hierarchy with leak-free grouping, proper cross-validation, and bootstrap confidence intervals.*

- **Action Items:**
  1. **Deterministic Group-Based Splitter:**
     - Stratified group splitting where `group = sentence_hash`. Ensures no duplicate or templated sentence ever leaks between train and test sets.
     - Random seeds strictly fixed and logged (`seed=42`).
  2. **4-Tier Model Ladder:**
     - **Level 1 (Traditional Baseline):** TF-IDF (character n-grams 2–4 + word n-grams 1–2) + Logistic Regression / Linear SVM.
     - **Level 2 (Static Embeddings):** Nepali fastText embeddings via `gensim` KeyedVectors + Logistic Regression (documenting bag-of-words limitations).
     - **Level 3 (Multilingual Transformer):** Fine-tuned `bert-base-multilingual-uncased` (mBERT).
     - **Level 4 (Indic Transformer):** Fine-tuned `google/muril-base-cased` and/or `NepBERTa` (`Rajan/NepBERTa`).
  3. **Mandatory Checkpoint Serialization:**
     - All models must save genuine serialized weights (`joblib.dump` for Level 1/2, `save_pretrained` for Level 3/4) into `checkpoints/<model_name>/`.
  4. **Statistical Evaluation Framework:**
     - Report **Macro-F1** as primary metric (never raw Accuracy).
     - Compute **95% Bootstrap Confidence Intervals** (1,000 resamples) for all test metrics.
     - Conduct McNemar's test for paired statistical significance between model tiers.
  5. **True Cross-Domain Generalization (Leave-One-Domain-Out / LODO):**
     - Protocol 1: In-domain 5-fold cross-validation.
     - Protocol 2: LODO transfer:
       - Train on News + Reviews $\rightarrow$ Test on Social Media.
       - Train on News + Social $\rightarrow$ Test on Reviews.
       - Train on Reviews + Social $\rightarrow$ Test on News.
       - Measure the cross-domain transfer penalty ($\Delta \text{F1}$).
- **Verification Gate:**
  - All benchmark results generated directly from verifiable checkpoint evaluation scripts; zero unverified target ranges.

---

### Phase 3.5: Nepali Linguistic Error Analysis & Subword Fertility Audit
*Goal: Provide deep linguistic insights into why models fail on low-resource Devanagari text.*

- **Action Items:**
  1. **Subword Fertility Analysis:**
     - Compute average subword tokens per word across Devanagari vocabulary for mBERT, MuRIL, and NepBERTa.
     - Quantify over-fragmentation and correlate fertility rate with classification error rate.
  2. **Morphological Negation Failure Analysis:**
     - Implement authentic Nepali morphological negation detector (`छैन`, `भएन`, `होइन`, `हुँदैन`, verbal prefix `न-`, e.g., `नगरेको`, `नआएको`).
     - Audit recall and precision on negative sentences containing subtle vs. explicit negation.
  3. **Qualitative Error Taxonomy:**
     - Categorize failure modes: Sarcasm/Irony, Named Entity Polarity Confusion, Mixed-Clause Dominance, Diacritic Stripping Artifacts.
- **Verification Gate:**
  - Publish `LINGUISTIC_ERROR_REPORT.md` with concrete confusion examples and fertility metrics.

---

### Phase 4.0: Inference Engine & Interactive Research Application
*Goal: Deliver a clean, reliable local inference interface and interactive verification application.*

- **Action Items:**
  1. **Decoupled Inference Engine (`nepali_sentiment/inference.py`):**
     - Clean Python API that loads verified saved checkpoints from `checkpoints/`.
     - Validates input Devanagari text, applies NFC normalization, outputs class probabilities and predicted sentiment.
  2. **Streamlit Research Demo (`app.py`):**
     - Interactive web UI allowing users to:
       - Test arbitrary Nepali sentences.
       - Inspect model confidence across classes.
       - View subword tokenization breakdown (revealing how transformers segment Devanagari words).
       - Select between Level 1 (TF-IDF), Level 2 (fastText), Level 3 (mBERT), and Level 4 (MuRIL).
- **Verification Gate:**
  - Automated smoke test verifying that `app.py` boots, loads saved weights, and returns valid predictions within <500ms on CPU.

---

### Phase 5.0: Research Documentation, Ethics & Open-Source Packaging
*Goal: Package the project for academic publication and open-source reproducibility.*

- **Action Items:**
  1. **Bender & Friedman Data Statement (`DATA_STATEMENT.md`):**
     - Speaker demographics, curation rationale, speech situations, annotator demographics.
  2. **Gebru et al. Datasheet (`DATASHEET.md`):**
     - Motivation, composition, collection process, preprocessing, recommended uses.
  3. **Research Ethics & Transparency Statement:**
     - Fair annotator compensation and consent.
     - Respect for platform terms of service and anonymization of personal identifiers.
  4. **Reproducibility Package:**
     - One-click reproduction script (`reproduce_benchmarks.sh` / `reproduce_benchmarks.ps1`).
     - Fixed seeds, pinned environment specs, public download links for checkpoints.
     - Comprehensive `README.md` reflecting actual verified numbers.
- **Verification Gate:**
  - Independent reproduction check on a clean environment from git clone to final metrics report.

---

## 3. Comparison of Roadmap Methodologies

| Dimension | Previous NepSent Scaffold | Revised Research Roadmap |
|---|---|---|
| **Data Provenance** | 100% synthetic (30 generator sentences). | Verified real-world collection across 3 distinct domains. |
| **Data Split Integrity** | Leaked identical sentences across train and test. | Strict group-aware hashing (`sha256(text)`). |
| **Annotation Process** | Pseudo-random dice simulator ($\kappa = -0.12$). | Double-blind human annotators with pilot calibration. |
| **Agreement Metrics** | Single unweighted $\kappa$ on random noise. | Unweighted $\kappa$, Linear-Weighted $\kappa_w$, and per-domain agreement. |
| **Disagreed Samples** | Merged without resolution (45.2% contradictions). | Structured consensus vs. adjudicated ambiguity partitions. |
| **Evaluation Metrics** | Accuracy / Macro-F1 on $N=17$ without variance. | Macro-F1 with 95% Bootstrap Confidence Intervals ($N \ge 300$). |
| **Cross-Domain Evaluation** | Slice on $N=6$ samples. | Rigorous Leave-One-Domain-Out (LODO) transfer protocol. |
| **Model Serialization** | Zero checkpoints saved (random head inference). | Standard Hugging Face / Joblib weight serialization. |
| **Linguistic Analysis** | `"न" in sentence` consonant bug. | Morphological negation parsing & subword fertility audit. |
| **Testing Lifecycle** | Deferred to final phase. | Integrated unit & integration tests from Phase 1. |

---

## 4. Single Highest-Priority Next Action

Before writing any new collection or modeling code, the **single highest-priority next action** is:

> **Execute Phase 0.5 (Repository Hygiene & Environment Security):**
> 1. Create a bulletproof `.gitignore` to prevent credential and binary leaks.
> 2. Add an official open-source `LICENSE` (Apache 2.0).
> 3. Relocate existing legacy data into `data/legacy_simulated/` to maintain the historical paper trail without corrupting new pipelines.
> 4. Centralize dynamic paths in `nepali_scraper/config.py` to eliminate all 14 hardcoded `d:/ds mdoel/` paths.

*We await your explicit review and approval before proceeding with Phase 0.5 implementation.*

