# PHASE ZERO: COMPLETE REPOSITORY AUDIT REPORT
**Project:** NepSent-Engine  
**Repository:** https://github.com/koiralapariksit/NepSent-Engine  
**Auditor:** Senior NLP Research Engineer & ML Systems Architect  
**Date:** November 3, 2026  
**Status:** Audit Completed — Awaiting User Approval Before Phase One Mutation  
**Core Benchmark Principle:** *Research integrity > reproducibility > data quality > model performance > UI polish*

---

## EXECUTIVE SUMMARY

NepSent-Engine was conceived as a low-resource NLP research system to benchmark sentiment classification across three domains (News & Politics, E-commerce Reviews, Social Media) using a four-tier model ladder (TF-IDF, fastText, mBERT, MuRIL).

Our exhaustive Phase Zero audit revealed a critical divergence between **the project's documentation claims** and **its underlying implementation**:
1. **Zero Genuine Scraped Data**: 100% of the existing corpus (all 60 records in `01_raw_corpus.jsonl`) originates from a hardcoded 30-sentence synthetic generator (`generate_test_data.py`). None of the 5 web scrapers have ever collected real-world data into the research corpus.
2. **Extreme Duplication & Leakage**: The 91 cleaned sentences contain only **31 unique sentences** (duplicated up to 6 times each). Due to un-deduplicated random partitioning, identical sentences exist simultaneously across the validation set, Annotator A training set, Annotator B training set, and the model test sets.
3. **Negative Cohen's Kappa ($\kappa = -0.1196$)**: Inter-annotator agreement was measured on simulated annotators who rolled pseudo-random dice (`random.choice([0, 1, 2])`) with different seeds (`seed=42` vs `seed=43`). Observed agreement (27.8%) was lower than expected agreement by chance (35.5%).
4. **Contradictory Training Labels**: Despite the negative Kappa score, annotations were blindly merged without adjudication into `04_annotated_dataset.jsonl`. Out of 31 unique sentences, **14 sentences (45.2%) have conflicting ground-truth labels** (e.g., the exact same sentence is labeled as Negative, Positive, AND Neutral in the same training set).
5. **Phantom Model Deployments & Unsaved Weights**: Zero model checkpoints exist on disk. `save_pretrained`, `torch.save`, and `joblib.dump` are never invoked in any training script. `classify_nepali.py` instantiates an untrained MuRIL model with a randomly initialized classification head.
6. **Hardcoded Environment Paths**: 14 files across the root directory hardcode the author's local development directory `d:/ds mdoel/data/`, which crashes all execution scripts on any newly cloned machine.
7. **Linguistic Bugs**: `phase3_error_analysis.py` contains a critical bug checking `"न" in sentence` for negation, incorrectly flagging almost every single sentence in the Nepali language containing the common consonant "न" (e.g. नेपाल, दिन, बनाउनु, गर्न).

---

## A. ARCHITECTURE AUDIT

### 1. Existing Architecture
The repository is split into two layers:
- `nepali_scraper/`: Python package containing `config.py`, `processors.py`, `scrapers.py`, `utils.py`.
- Root scripts: Disjoint scripts representing Phases 1, 2, and 3 (`scraper_main.py`, `process_pipeline.py`, `phase2_*.py`, `phase3_*.py`, `classify_nepali.py`).

```
d:\Nepsent-Engine\
├── nepali_scraper/             # Core scraping package
│   ├── config.py               # Settings (defines PROJECT_ROOT dynamically)
│   ├── processors.py           # NepaliTextProcessor (NFC, regex filtering)
│   ├── scrapers.py             # BaseScraper, Ratopati, Setopati, Onlinekhabar, Daraz, Twitter
│   └── utils.py                # Logging, validation, reports
├── data/                       # 14 data & result files (all synthetic / simulated)
├── generate_test_data.py       # Hardcoded 30-sample synthetic data generator
├── process_pipeline.py         # Pipeline runner
├── phase2_*.py                 # Splitter, simulator, kappa calculator, merger, orchestrator
├── phase3_*.py                 # TFIDF, fastText, mBERT, MuRIL, error analysis, orchestrator
└── classify_nepali.py          # Demo script (evaluates untrained MuRIL)
```

### 2. Component Status Analysis

| Component | File(s) | Status | Key Flaws / Failure Modes |
|---|---|---|---|
| **Configuration** | `nepali_scraper/config.py` | Partially Working | Defines `PROJECT_ROOT`, but prints to stdout on every import. Root scripts ignore this file and hardcode `d:/ds mdoel/data`. |
| **Text Processing** | `nepali_scraper/processors.py` | Flawed | `NepaliTextProcessor` drops any sentence without a terminal delimiter (`।`, `?`, `!`). Paragraph segmentation breaks on consecutive delimiters (e.g., `!!`, `?...`). Destructive regex strips em-dashes and quotes. |
| **Web Scrapers** | `nepali_scraper/scrapers.py` | Broken / Inoperable | • **Ratopati**: CSS selectors outdated.<br>• **Setopati**: Timeouts / anti-bot blocking.<br>• **Onlinekhabar**: Invalid pagination endpoint (`/news?page=`).<br>• **Daraz**: Static BeautifulSoup fails on dynamic CSR JavaScript apps.<br>• **Twitter**: Paid API v2 Bearer token required ($100/mo minimum). |
| **Annotation Engine** | `phase2_*.py` | Incomplete / Synthetic | No human UI (no Streamlit). Relies on `phase2_annotation_simulator.py`. Merge logic combines conflicting labels without adjudication. |
| **Statistical Agreement** | `phase2_kappa_calculation.py` | Functioning (Math) | Math is correct (`sklearn.metrics.cohen_kappa_score`), but input data is random simulation, resulting in $\kappa = -0.1196$. |
| **Model Trainers** | `phase3_*.py` | Severe Leakage / No Checkpoints | `BaseModelTrainer` splits duplicated sentences across train/test. Models are trained on contradictory labels. Zero weights saved to disk. |
| **Inference Application** | `classify_nepali.py` | Broken / Fake | Loads raw Hugging Face backbone with random classification head. Never loads trained weights. |
| **Error Analysis** | `phase3_error_analysis.py` | Linguistically Invalid | Checks `"न" in sentence`, `"नै" in sentence` (emphatic, not negative), and Hindi `"नहीं"` for Nepali negation. |

### 3. Duplicated and Inconsistent Components
- `phase2_data_split.py` and `phase3_model_trainer.py` both implement separate splitting logic without shared seed control.
- `PHASE2_EXECUTION_GUIDE.py` and `phase2_orchestrator.py` duplicate execution instructions and CLI handlers.
- `PHASE3_EXECUTION_GUIDE.py` and `phase3_orchestrator.py` duplicate benchmark targets and descriptions.
- `data_dir` is redundantly declared across 14 independent scripts instead of importing from `nepali_scraper.config`.

### 4. Dependency and Environment Problems
- Missing `pytest`, `streamlit`, `mypy`, `black`, `flake8` in `requirements.txt`.
- `phase3_fasttext_model.py` runs `os.system("pip install fasttext")` at runtime. On Windows, `fasttext` fails to compile without MSVC C++ Build Tools.
- `torch` is specified as `torch>=2.0.0` without CUDA/CPU wheel distinction, triggering a 130MB–2.5GB download during setup.
- Multiple scripts call `import subprocess; subprocess.check_call(["pip", "install", ...])` dynamically inside module imports, violating environment hygiene.

### 5. Security and Confidentiality
- `.env.example` exists, but `.env` was created locally. `.gitignore` must explicitly prevent committing `.env`, `*.bin`, `*.pt`, and API tokens.
- No credential leakage detected in git history, but scraping scripts lack proper TLS verification parameterization and User-Agent rotation.

### 6. Scalability Limitations
- Scrapers use synchronous, single-threaded requests with naive `time.sleep()`.
- Data storage relies on unindexed JSONL files loaded entirely into memory (`f.readlines()`).
- Transformers (`mBERT`, `MuRIL`) are fine-tuned using naive PyTorch loops without gradient accumulation, FP16/BF16 mixed precision, or evaluation checkpoints.

---

## B. DATASET AUDIT

We performed a deep automated inspection of all dataset files in `data/`.

### 1. Dataset Provenance Breakdown

| File | Total Records | Unique Sentences | Provenance Classification | Genuine Real-World | Human Annotated |
|---|---|---|---|---|---|
| `01_raw_corpus.jsonl` | 60 | 21 | `SYNTHETIC_TEST` | **0 (0.0%)** | 0 |
| `02_cleaned_corpus.jsonl` | 91 | 31 | `SYNTHETIC_TEST` | **0 (0.0%)** | 0 |
| `03_validation_partition.jsonl` | 36 | 13 | `SIMULATED_ANNOTATION` | **0 (0.0%)** | 0 |
| `03_training_annotator_A.jsonl` | 36 | 17 | `SIMULATED_ANNOTATION` | **0 (0.0%)** | 0 |
| `03_training_annotator_B.jsonl` | 37 | 19 | `SIMULATED_ANNOTATION` | **0 (0.0%)** | 0 |
| `04_annotated_dataset.jsonl` | 109 | 31 | `SIMULATED_ANNOTATION` | **0 (0.0%)** | 0 |

### 2. Domain & Platform Distribution in Existing Files

```
Raw Corpus (01_raw_corpus.jsonl - 60 records):
  ├── news_politics      : 30 (50.0%) [Ratopati: 10, Setopati: 10, Onlinekhabar: 10]
  ├── ecommerce_reviews  : 20 (33.3%) [Daraz: 20]
  └── social_media       : 10 (16.7%) [Twitter: 10]

Cleaned Corpus (02_cleaned_corpus.jsonl - 91 sentences):
  ├── news_politics      : 60 (65.9%)
  ├── ecommerce_reviews  : 20 (22.0%)
  └── social_media       : 11 (12.1%)

Annotated Dataset (04_annotated_dataset.jsonl - 109 records):
  ├── news_politics      : 68 (62.4%)
  ├── ecommerce_reviews  : 27 (24.8%)
  └── social_media       : 14 (12.8%)
```

### 3. Duplicate Analysis & Contamination
- In `01_raw_corpus.jsonl`, only 21 unique texts exist. The 10 synthetic news texts were cloned across Ratopati, Setopati, and Onlinekhabar. The 10 synthetic reviews were duplicated 2x for Daraz.
- In `02_cleaned_corpus.jsonl`, the top 5 sentences appear **6 times each**:
  1. `नेपालको अर्थनीति यस वर्ष बढदारी गरिरहेको छ।` (Count: 6)
  2. `काठमाडौंमा आज सबेरे मूसलधार वर्षा भएको छ।` (Count: 6)
  3. `सरकारले नयाँ शिक्षा नीति घोषणा गरेको छ।` (Count: 6)
  4. `भारत र नेपालबीच द्विपक्षीय वार्ता सफल भएको छ।` (Count: 6)
  5. `गत महिना विदेशी पर्यटकको आगमन बढेको छ।` (Count: 6)
- Because duplicates were never removed before partitioning:
  - Validation set has 13 unique sentences.
  - **5 of these sentences also appear in Training Annotator A.**
  - **7 of these sentences also appear in Training Annotator B.**
  - **9 sentences overlap between Training Annotator A and B.**

### 4. Ground-Truth Contradictions in Annotated Dataset
In `04_annotated_dataset.jsonl`, **14 out of 31 unique sentences (45.2%) have conflicting labels**:
- `"गत महिना विदेशी पर्यटकको आगमन बढेको छ।"`: Labeled **0 (Negative)**, **1 (Positive)**, AND **2 (Neutral)**.
- `"काठमाडौंमा आज सबेरे मूसलधार वर्षा भएको छ।"`: Labeled **0 (Negative)** AND **2 (Neutral)**.
- `"नेपमा सामाजिक माध्यम प्रयोगकर्ता संख्या बढ्दै छ।"`: Labeled **0**, **1**, AND **2** across annotators.
- `"यो किताब पढ्न बहुत रोचक छ।"`: Labeled **0 (Negative)** AND **1 (Positive)**.

---

## C. ANNOTATION AUDIT

### 1. Guidelines & Label Schema
- Guidelines in `phase2_annotation_guidelines.py` specify a 3-class schema: `0: Negative`, `1: Positive`, `2: Neutral`.
- Three heuristic rules are described:
  1. *Sarcasm Law*: Intent over literal words.
  2. *Final-Clause Rule*: In compound sentences, prioritize the terminal clause.
  3. *Objective Boundary*: Factual news without evaluative adjectives is Neutral.
- **Flaw**: The guidelines are only present as Python docstrings. No human interface or annotation portal exists.

### 2. Annotator Assignment & Simulation
- Annotator assignments were purely theoretical. In practice, `phase2_annotation_simulator.py` simulated Annotator A and Annotator B using hardcoded wordlists and random fallback:
  ```python
  if domain == "news_politics":
      if neg_count > 0 and pos_count == 0:
          return 0
      elif pos_count > 0 and neg_count == 0:
          return 1
      else:
          return random.choice([0, 1, 2]) if random.random() < 0.7 else 2
  ```
- Annotator A was assigned seed 42.
- Annotator B was assigned seed 43.
- Because the synthetic sentences lacked the specific words in `negative_words` or `positive_words`, almost every sentence triggered `random.choice([0, 1, 2])`.

### 3. Investigation of the Negative Cohen's Kappa
In `data/07_kappa_report.json`:
- Total validation sentence pairs: 18
- Number of agreed pairs: 5 (27.8%)
- Number of disagreed pairs: 13 (72.2%)
- Confusion Matrix:
  $$\begin{pmatrix} 0 & 1 & 4 \\ 4 & 5 & 1 \\ 2 & 1 & 0 \end{pmatrix}$$
- Marginal distributions:
  - Annotator A: Neg = 5, Pos = 10, Neu = 3
  - Annotator B: Neg = 6, Pos = 7, Neu = 5
- Observed agreement $P_o = \frac{5}{18} \approx 0.2778$.
- Expected agreement by chance:
  $$P_e = \frac{5 \times 6 + 10 \times 7 + 3 \times 5}{18^2} = \frac{30 + 70 + 15}{324} = \frac{115}{324} \approx 0.3549$$
- Cohen's Kappa calculation:
  $$\kappa = \frac{P_o - P_e}{1 - P_e} = \frac{0.2778 - 0.3549}{1 - 0.3549} = \frac{-0.0771}{0.6451} \approx -0.1196$$

**Mathematical Root Cause**:
Cohen's Kappa was negative because Annotators A and B were pseudo-random generators seeded with different seeds. The observed agreement rate ($27.8\%$) was strictly lower than chance expectation ($35.5\%$).
There was zero human agreement measured. The reported agreement represents simulated stochastic noise.

---

## D. MODEL AUDIT

### 1. Four-Tier Model Ladder Evaluation

| Level | Model | Target Macro F1 | Reported Macro F1 | Reported Accuracy | Evaluation Test Set Size | Status |
|---|---|---|---|---|---|---|
| **Level 1** | TF-IDF + Logistic Regression | 55–65% | 38.9% | 41.2% | $N = 17$ | Failed target; evaluated on contaminated split |
| **Level 2** | fastText + Logistic Regression | 63–72% | 45.7% | 47.1% | $N = 17$ | Evaluated on contaminated split; hardcoded model path |
| **Level 3** | fine-tuned mBERT | 74–82% | 53.1% | 58.8% | $N = 17$ | Severe overfitting on duplicate records; no checkpoint |
| **Level 4** | fine-tuned MuRIL | 79–86% | 51.1% | 52.9% | $N = 17$ | Underperformed mBERT; no saved checkpoint |

### 2. Methodological Failures in Modeling Pipeline
1. **Extreme Data Leakage**:
   `BaseModelTrainer.split_train_test()` performs a naive stratified train/test split on the 109 rows of `04_annotated_dataset.jsonl`. Because the dataset contains only 31 unique sentences, identical sentences are placed in both the train and test sets.
2. **Evaluation on 17 Samples**:
   The test set consists of exactly 17 sentences. A single misclassification swings accuracy and F1 by nearly $6\%$. Any metric computed on $N=17$ is statistically meaningless and carries a massive confidence interval ($\pm 25\%$).
3. **Training on Mutually Contradictory Data**:
   The models were trained on data where 45.2% of sentences had contradictory labels. A model cannot learn a coherent decision boundary when the exact same token sequence is labeled as 0, 1, and 2.
4. **No Model Checkpoints**:
   None of the training scripts serialize the trained parameters. When `classify_nepali.py` runs, it calls `MuRILModel(dataset_path)` which loads raw Hugging Face weights with an un-finetuned, randomly initialized classifier head.

---

## E. RESEARCH VALIDITY ASSESSMENT

We classify every historical artifact in the repository under strict empirical standards:

| Artifact / Result | Claimed Status | Verified Research Classification | Justification |
|---|---|---|---|
| `data/01_raw_corpus.jsonl` | "Scraped multi-domain corpus" | **DEMONSTRATION ONLY (`SYNTHETIC_TEST`)** | Generated by `generate_test_data.py`. No web requests were saved. |
| `data/02_cleaned_corpus.jsonl` | "Processed Devanagari corpus" | **DEMONSTRATION ONLY (`SYNTHETIC_TEST`)** | Derived directly from synthetic corpus. 31 unique sentences. |
| `data/03_validation_partition.jsonl` | "Dual annotator validation set" | **DEMONSTRATION ONLY (`SIMULATED_ANNOTATION`)** | Labels generated by heuristic random simulation. |
| `data/07_kappa_report.json` | "Inter-annotator agreement report" | **INVALID / METHODOLOGICALLY UNRELIABLE** | Measures disagreement between two random seeds ($\kappa = -0.12$). |
| `data/04_annotated_dataset.jsonl` | "Final annotated dataset" | **INVALID / METHODOLOGICALLY UNRELIABLE** | Contains 14 sentences with self-contradictory labels. |
| `data/08a_tfidf_results.json` | "TF-IDF baseline benchmark" | **INVALID / METHODOLOGICALLY UNRELIABLE** | Evaluated on $N=17$ test set with severe train/test sentence leakage. |
| `data/08b_fasttext_results.json` | "fastText benchmark" | **INVALID / METHODOLOGICALLY UNRELIABLE** | Evaluated on leaked $N=17$ test set. |
| `data/08c_mbert_results.json` | "mBERT benchmark" | **INVALID / METHODOLOGICALLY UNRELIABLE** | Evaluated on leaked $N=17$ test set; no saved weights. |
| `data/08d_muril_results.json` | "MuRIL benchmark" | **INVALID / METHODOLOGICALLY UNRELIABLE** | Evaluated on leaked $N=17$ test set; no saved weights. |
| `data/09_model_comparison.json` | "Cross-model comparison rankings" | **INVALID / METHODOLOGICALLY UNRELIABLE** | Ranks models based on leaked, unverified 17-sample evaluations. |
| `classify_nepali.py` | "Production Nepali sentiment inference" | **BROKEN / DEMONSTRATION ONLY** | Evaluates randomly initialized PyTorch classification head. |

---

## F. PRIORITIZED UPGRADE ROADMAP

| ID | Issue | Severity | Technical Explanation | Recommended Fix | Complexity |
|---|---|---|---|---|---|
| **R-01** | Hardcoded `d:/ds mdoel` Paths | **CRITICAL** | 14 root scripts crash on any machine where the repo is not cloned to `d:\ds mdoel`. | Replace all hardcoded paths with centralized dynamic paths from `nepali_scraper.config`. | Low |
| **R-02** | Contradictory & Leaked Dataset | **CRITICAL** | 45.2% of annotated sentences have contradictory labels; train/test splits share identical sentences. | Enforce strict sentence deduplication, provenance tracking, and group-aware/hash-based splitting. | Medium |
| **R-03** | Broken Web Scraping Engine | **CRITICAL** | Scrapers fail due to outdated selectors, dynamic JS rendering on Daraz, and paid Twitter API constraints. | Implement working collectors for accessible Nepali news sources (e.g. RSS/API/validated HTML) and public datasets; document authorized collection policies. | High |
| **R-04** | Missing Human Annotation UI & Workflow | **CRITICAL** | No human annotation interface exists; annotations were random simulations resulting in $\kappa = -0.12$. | Build a Streamlit annotation app with independent dual-annotator blinding, consensus adjudication, and honest Kappa tracking. | High |
| **R-05** | No Model Serialization & Broken Inference | **CRITICAL** | Zero trained model weights are saved; inference script runs randomly initialized heads. | Implement standard checkpoint saving (`save_pretrained`, `joblib.dump`) and a decoupled inference engine. | High |
| **R-06** | Text Processor Delimiter Dropping | **HIGH** | Sentences without trailing `।`, `?`, `!` are silently dropped. Punctuation regex mangles quotes and dashes. | Overhaul `NepaliTextProcessor` to retain terminal boundary-less sentences, normalize NFC, and preserve code-mixed text. | Medium |
| **R-07** | Inverted Linguistic Error Analysis | **HIGH** | `"न" in sentence` marks every sentence containing the letter "न" as Negation; `"नै"` is emphatic, not negative. | Rewrite error analyzer with authentic Nepali morphological negation rules (`होइन`, `छैन`, `भएन`, verb suffixes). | Medium |
| **R-08** | Inadequate Environment & Dependency Hygiene | **HIGH** | `requirements.txt` lacks testing and UI tools; runtime `pip install` commands pollute execution. | Clean `requirements.txt`, pin dependencies, add `pytest` suite and GitHub Actions CI. | Medium |
| **R-09** | Inflationary Claims in Documentation | **MEDIUM** | README claims 4,000–5,000 verified sentences and production status when only 31 synthetic sentences exist. | Align all documentation with verified implementation reality. | Low |

---

## G. SUMMARY FOR USER APPROVAL

We have completed Phase Zero with full adherence to the **Gravitas Invariants** and research integrity principles.

### Required Actions Before Phase One:
1. **Preserve Historical Artifacts**: Move existing simulated data to `data/legacy_simulated/` to maintain a historical paper trail without corrupting new pipelines.
2. **Centralize Paths**: Eliminate all `d:/ds mdoel` references.
3. **Approve Phase One Implementation**: We will await your formal approval before initiating Phase One (Data Engineering & Dataset Integrity).

