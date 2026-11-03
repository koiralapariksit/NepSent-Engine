# NepSent-Engine: Comprehensive Research & Operational Risk Register

**Document:** `RISK_REGISTER.md`  
**Status:** Living Engineering & Methodological Risk Control Matrix  
**Last Updated:** November 2026  
**Operating Principle:** *Identify failure modes proactively; design verified fallback pathways.*

---

## 1. Risk Matrix Overview

```
CRITICAL (Red)   │ R-01 (Data Scraping Access)   R-03 (Sample Size & Overfitting)  R-07 (Security & .gitignore)
HIGH (Orange)    │ R-02 (Annotator Reliability)  R-04 (Domain Class Imbalance)    R-06 (Windows / C++ Tooling)
MEDIUM (Yellow)  │ R-05 (Linguistic Ambiguity)   R-08 (Inference Latency on CPU)  R-09 (Scope Creep)
                 └────────────────────────────────────────────────────────────────────────────────────────
                   LOW LIKELIHOOD               MEDIUM LIKELIHOOD                HIGH LIKELIHOOD
```

---

## 2. Detailed Risk Assessment and Mitigation

### R-01: Web Scraping Legal, Ethical, and Technical Blockers
- **Category:** Data Engineering / Ethical Compliance
- **Severity / Likelihood:** **CRITICAL / HIGH**
- **Trigger:** News websites update anti-scraping Cloudflare protections; Daraz employs dynamic client-side rendering (CSR); Twitter API v2 enforces prohibitive $100+/month tiers; rate limits or ToS violations.
- **Impact:** Inability to collect genuine raw text in target domains, halting the pipeline.
- **Prevention & Mitigation:**
  1. **Do not fight dynamic anti-bot systems:** Replace fragile headless browser scraping with public, authorized data sources:
     - News: Use official RSS feeds and public sitemaps (e.g., Ratopati and Setopati RSS endpoints provide clean Devanagari text without bot challenges).
     - E-commerce: Scrape public reviews only from static HTML stores or utilize open-source Indic product review datasets (e.g., from Hugging Face / Kaggle) with documented attribution.
     - Social Media: Utilize public academic dumps (e.g., NepCOV19Tweets or Nepali Reddit public dumps from Sitoula et al., 2025) rather than attempting unauthorized Twitter scraping.
  2. Implement strict politeness policies: randomized delays (2.0–5.0 seconds), custom user-agent headers identifying the academic project, and zero CAPTCHA bypassing.
- **Fallback Plan:** Ingest verified public Nepali corpora under permissive licenses (`EXTERNAL_VERIFIED_BENCHMARK`) to fulfill domain quotas if live scraping is throttled.

---

### R-02: Human Annotator Reliability, Agreement Failure, and Cognitive Fatigue
- **Category:** Annotation Methodology
- **Severity / Likelihood:** **HIGH / HIGH**
- **Trigger:** Volunteer or student annotators exhibit low agreement ($\kappa < 0.40$), misunderstand guidelines, or suffer cognitive fatigue after labeling >200 sentences.
- **Impact:** Unreliable ground truth labels; negative or insignificant Kappa scores; invalidation of benchmark.
- **Prevention & Mitigation:**
  1. **Pilot Calibration Gate (Phase 2):** Require two independent annotators to label an initial calibration batch of 30–50 sentences.
  2. **Guideline Refinement:** Conduct an interactive adjudication meeting after the pilot to discuss disagreements, disambiguate borderline categories (e.g., factual reporting vs. neutral opinion), and refine docstrings and examples.
  3. **Batch Limiting:** Cap annotator sessions at 100 sentences per sitting with mandatory breaks to combat fatigue.
  4. **Ordinal Weighting:** Measure and optimize for linear-weighted $\kappa_w$, which penalizes opposite-polarity errors (Positive vs. Negative) more heavily than adjacent errors (Neutral vs. Positive).
- **Fallback Plan:** If agreement on the 3-class schema remains poor ($\kappa < 0.50$), collapse to a binary polarity benchmark (Positive vs. Negative, excluding ambiguous Neutral) as a documented methodological fallback.

---

### R-03: Statistical Power Deficit & Transformer Overfitting on Small Datasets
- **Category:** Machine Learning / Statistical Rigor
- **Severity / Likelihood:** **CRITICAL / MEDIUM**
- **Trigger:** Fine-tuning massive models (mBERT: 110M params, MuRIL: 237M params) on a small training partition ($N < 1,000$) causes rapid memorization, catastrophic forgetting, or high evaluation variance.
- **Impact:** Overfitting on training duplicates; wildly swinging F1 scores; non-generalizable checkpoints.
- **Prevention & Mitigation:**
  1. **Layer Freezing / Parameter-Efficient Fine-Tuning (PEFT):** Freeze the lower 8–10 transformer encoder layers and train only the classification head and top 2 transformer layers.
  2. **Aggressive Regularization:** Apply weight decay ($0.01$), dropout ($0.2–0.3$), and early stopping with patience = 2 epochs on the validation loss.
  3. **Leakage-Free Sentence Hashing:** Guarantee that identical sentences cannot appear in both train and test sets.
  4. **Statistical Power Standards:** Set target test partition size to $N \ge 300$ (minimum 100 samples per class) to ensure the 95% bootstrap confidence interval width remains under $\pm 6\%$.
- **Fallback Plan:** If transformer fine-tuning exhibits high variance, report frozen-feature extraction (extracting `[CLS]` embeddings into an SVM/Logistic Regression classifier) as the primary neural baseline alongside TF-IDF.

---

### R-04: Natural Domain Class Imbalance
- **Category:** Data Distribution / Evaluation
- **Severity / Likelihood:** **HIGH / HIGH**
- **Trigger:** News text naturally skews >70% Neutral; e-commerce reviews naturally skew >60% Positive; social media text skews Negative or sarcastic.
- **Impact:** Models default to majority-class prediction; standard Accuracy appears high (e.g., 75%) while Macro-F1 collapses (e.g., 35%).
- **Prevention & Mitigation:**
  1. **Stratified Sampling:** Enforce domain-stratified and class-stratified splits during dataset assembly.
  2. **Macro-Averaging as Primary Metric:** Never use Accuracy or Micro-F1 as the ranking metric; report **Macro-F1** and balanced accuracy.
  3. **Class-Weighted Loss:** Weight the cross-entropy loss inversely proportional to class frequencies during training:
     $$w_c = \frac{N}{3 \cdot N_c}$$
- **Fallback Plan:** Apply focal loss or balanced undersampling of over-represented Neutral sentences in the news partition if training divergence occurs.

---

### R-05: Linguistic Complexity (Code-Switching, Sarcasm, Agglutinative Negation)
- **Category:** NLP Modeling
- **Severity / Likelihood:** **MEDIUM / HIGH**
- **Trigger:** Social media data contains English words in Roman script or Devanagari transliteration; sarcasm reverses literal meaning; Nepali negation prefixes (`न-`) are missed by bag-of-words or broken tokenizers.
- **Impact:** Misclassification of inverted sentiment; low recall on negative samples.
- **Prevention & Mitigation:**
  1. **Preserve Code-Mixed Text:** Do not filter out Romanized or English tokens; let subword tokenizers process them natively.
  2. **Audited Devanagari Normalization:** Ensure Unicode NFC normalization preserves virama (`्`) and nukta characters, as recommended by Baral et al. (2026).
  3. **Dedicated Negation Lexicon & Morphology Checker:** Build a linguistically sound negation detector recognizing `छैन`, `भएन`, `होइन`, `हुँदैन`, and verbal prefix `न-` for the error analysis suite.
- **Fallback Plan:** Tag code-switched and sarcastic samples in metadata to isolate their impact during error analysis.

---

### R-06: Windows Environment & Native Compilation Tooling (fastText / PyTorch)
- **Category:** Software Engineering / Reproducibility
- **Severity / Likelihood:** **HIGH / MEDIUM**
- **Trigger:** The official `fasttext` package requires Microsoft Visual C++ Build Tools on Windows to compile native C++ extensions, causing `pip install fasttext` to fail. PyTorch downloads can stall or default to massive CUDA wheels on CPU machines.
- **Impact:** Broken automated setups; contributors unable to run the Level 2 model on Windows.
- **Prevention & Mitigation:**
  1. **Pre-compiled Wheels / Alternative Packages:** Use `fasttext-wheel` or load pre-trained Nepali fastText vectors (`cc.ne.300.vec.gz`) directly via `gensim.models.KeyedVectors`, which does not require MSVC compilation.
  2. **CPU-Specific Torch Wheels:** Provide explicit instructions and flags in `requirements.txt` for CPU vs. CUDA installations.
  3. **Eliminate Runtime `os.system("pip install ...")`:** All dependencies must be pre-declared and verified in `requirements.txt`.
- **Fallback Plan:** Provide `gensim`-based fastText loading as the default fallback in `phase3_fasttext_model.py`.

---

### R-07: Security, Credential Exposure, and Repository Hygiene
- **Category:** Information Security / Open-Source Maintenance
- **Severity / Likelihood:** **CRITICAL / LOW (Post-Mitigation)**
- **Trigger:** Absence of `.gitignore` causes accidental commit of `.env`, API tokens, virtual environments, or large model binaries.
- **Impact:** Revocation of API keys; bloated repository (>1GB); copyright violations.
- **Prevention & Mitigation:**
  1. **Immediate `.gitignore` Deployment:** Add a comprehensive `.gitignore` covering `.env`, `venv/`, `__pycache__/`, `*.pt`, `*.bin`, `*.safetensors`, and data scratch files before any git commits are staged.
  2. **Add Standard Open-Source License:** Add `LICENSE` (Apache 2.0 or MIT for code; CC BY-SA 4.0 for dataset).
  3. **Secrets Scanning:** Run automated checks to ensure no bearer tokens or hardcoded paths exist in committed code.
- **Fallback Plan:** If credentials are ever committed, immediately revoke tokens, rewrite history via `git-filter-repo`, and notify stakeholders.

---

### R-08: Model Training / Inference Latency on Commodity Hardware
- **Category:** Computational Feasibility
- **Severity / Likelihood:** **MEDIUM / MEDIUM**
- **Trigger:** Fine-tuning mBERT or MuRIL on CPU machines takes hours per epoch, impeding rapid iteration.
- **Impact:** Developer friction; stalled test cycles.
- **Prevention & Mitigation:**
  1. **Small Batch Sizes with Gradient Accumulation:** Use batch size 8–16 with accumulation steps.
  2. **Lightweight Baselines First:** Fast turnaround on TF-IDF (<10s) and fastText (<1m) allows verifying dataset and pipeline integrity before touching transformers.
  3. **Subset Evaluation during Development:** Unit tests and CI pipelines must run against a tiny mock dataset ($N=10$) with `max_iter=1` or tiny models.
- **Fallback Plan:** Utilize free Google Colab / Kaggle GPU runtimes for final transformer fine-tuning, storing only model checkpoints in the repo.

---

### R-09: Scope Creep and Inflation of Claims
- **Category:** Research Integrity
- **Severity / Likelihood:** **MEDIUM / MEDIUM**
- **Trigger:** Temptation to expand into 10 domains, multimodal sentiment, or claims of superhuman accuracy before completing core baselines.
- **Impact:** Incomplete releases; unverified assertions; desk rejection.
- **Prevention & Mitigation:**
  1. **Strict 3-Domain Boundary:** Lock domains strictly to News/Politics, E-Commerce Reviews, and Social Media.
  2. **Strict 4-Tier Model Ladder:** Lock models strictly to TF-IDF, fastText, mBERT, and MuRIL (or NepBERTa).
  3. **Enforce Gravitas Invariants:** Never report unverified metrics; preserve historical artifacts; back every claim with empirical logs.
- **Fallback Plan:** Defer multimodal, aspect-based, or fine-grained 5-point sentiment tasks to post-publication future work.

