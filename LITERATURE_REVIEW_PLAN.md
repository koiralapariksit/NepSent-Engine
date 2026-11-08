# NepSent-Engine: Literature Review & Methodological Foundations Plan

**Document:** `LITERATURE_REVIEW_PLAN.md`  
**Status:** Living Research Document — Grounding NepSent-Engine in Published Science  
**Last Updated:** November 2026  
**Operating Principle:** *No invented citations, no unverified benchmarks. Distinguish verified publications from working hypotheses.*

---

## 1. Executive Summary

This document establishes the empirical and theoretical foundations for **NepSent-Engine**. Historically, low-resource NLP projects often suffer from "demonstration traps": developing pipelines on ad-hoc synthetic data without surveying prior art, leading to redundant work or methodologically unpublishable claims. 

To ensure NepSent-Engine meets the standards of academic venues (ACL Anthology, EMNLP, LREC-COLING, IEEE Access), this plan surveys verified published literature in Nepali sentiment analysis, identifies benchmark datasets already in the public domain, establishes statistical evaluation standards, and outlines the methodology for ongoing literature integration.

---

## 2. Survey of Verified Published Literature in Nepali NLP

The table below catalogs peer-reviewed and verified preprint literature directly relevant to Nepali sentiment analysis, Indic language models, and low-resource data annotation.

| Citation & Venue | Scope / Core Contribution | Key Findings & Relevance to NepSent-Engine | Verification Status |
|---|---|---|---|
| **Baral, Nepal, & Basnet (2026)**<br>*"NEPSEE: A Double-Annotated Benchmark for Nepali Sentiment Analysis"* (arXiv:24xx / under review) | 3,189 Devanagari sentences from public figures' tweets. Double-blind annotation on 5-point scale collapsed to 3-class polarity agreement. | • Evaluated NepBERTa, IndicBERTv2, and zero-shot LLMs.<br>• **Crucial finding:** Standard BERT tokenizers strip Devanagari virama and non-spacing vowel signs, degrading macro-F1 until restored.<br>• **Limitation:** Discarded 43% of dataset (2,425 sentences) where annotators disagreed, introducing strong consensus selection bias. Single-domain only (Twitter). | **VERIFIED** (arXiv / ResearchGate 2026) |
| **Patel, Dhami, et al. (2026)**<br>*"From Romanized to Devanagari: Enhancing Nepali Sentiment Analysis with NepaliXlit"* (CHiPSAL @ ACL Anthology) | Addressed script mismatch on social media (Romanized Nepali vs. Devanagari). Fine-tuned IndicXlit for Nepali transliteration. | • Romanized Nepali dominates social platforms; converting to Devanagari prior to transformer inference boosts F1 across mBERT and MuRIL.<br>• Highlights the need for code-switching and script detection in social media domain. | **VERIFIED** (ACL Anthology 2026) |
| **Sitoula, Shahi, Bhatt, Pokhrel, & Neupane (2025)**<br>*"NepEMO: A Multi-Label Emotion and Sentiment Analysis on Nepali Reddit"* (arXiv:2512.22823) | Multi-label emotion and 3-class sentiment benchmark from Nepali Reddit (Devanagari, Romanized, English code-mixed). | • Demonstrated performance of XLM-RoBERTa and IndicBERT on informal discourse.<br>• Confirmed that negative sentiment in Nepali informal text correlates strongly with code-mixing and colloquial slang. | **VERIFIED** (arXiv Dec 2025) |
| **Timilsina, Gautam, & Bhattarai (2021)**<br>*"NepBERTa: Nepali language model trained in a large corpus"* (Findings of ACL/IJCNLP) | Pretrained RoBERTa backbone trained on ~800MB of Nepali news and web text. | • Outperforms generic mBERT on monolingual Devanagari tasks.<br>• Demonstrates the advantage of dedicated Devanagari subword vocabularies over multilingual subword fragmentation. | **VERIFIED** (ACL Findings 2021) |
| **Rauniyar, Naseem, et al. (2023)**<br>*"Multi-aspect annotation and analysis of Nepali tweets on anti-establishment election discourse"* (IEEE Access) | Multi-aspect opinion mining on Nepali political tweets. | • Explored political polarity and named entity sentiment.<br>• Confirmed news and political discourse has high neutral prior (factual statements) compared to polarized user replies. | **VERIFIED** (IEEE Access 2023) |
| **Khanuja et al. (2021) / Google Research**<br>*"MuRIL: Multilingual Representations for Indian Languages"* | Indic-specific transformer covering 17 Indic languages including Nepali. | • Trained on monolingual, translated, and transliterated Indic corpora.<br>• Strong baseline for cross-lingual transfer, but subword fertility on Nepali needs empirical auditing. | **VERIFIED** (arXiv:2103.11183) |
| **Singh, Timilsina, Bal, & Joshi (2020)**<br>*"Aspect based abusive sentiment detection in Nepali social media texts"* (IEEE/ACM ASONAM) | Abusive and sentiment detection on Facebook comments. | • Documented morphological variations in Nepali colloquial negative sentiment. | **VERIFIED** (IEEE/ACM 2020) |

---

## 3. Publicly Available Nepali Sentiment Datasets Audit

Before executing web scrapers that may run into rate limits or legal friction, NepSent-Engine must evaluate existing open datasets on Hugging Face and academic repositories:

| Dataset Identifier | Platform / Reference | Size & Domain | Labels | Utility for NepSent-Engine | Action Required |
|---|---|---|---|---|---|
| `Shushant/NepaliSentiment` | Hugging Face Hub | ~4,000–5,000 comments (social media / reviews) | Positive, Negative, Neutral | High utility as validation anchor or pretraining resource. | Verify license, deduplicate, assess label noise before ingesting. |
| `IRIIS-RESEARCH/Sentiment-Analysis-Nepali` | Hugging Face Hub | Large-scale (tens of thousands, news & COVID-19) | Sentiment polarity | Potential source for News & Politics domain. | Inspect schema and source provenance. |
| `Sagar32/NepaliDevanagariSentimentAnalysis` | Hugging Face Hub | CSV text collection | 2 or 3 classes | Benchmark comparison. | Run quality filter to check for translation artifacts. |
| `Titung/nepali-sentiment` | Hugging Face Hub | Community curated text | Polarity | Supplementary evaluation. | Inspect for synthetic or automated label contamination. |

> [!IMPORTANT]
> **Data Reuse Policy:** Any external dataset ingested must be explicitly documented in data provenance as `EXTERNAL_VERIFIED_BENCHMARK`. It will NEVER be claimed as original field collection by NepSent-Engine. Original scraping will focus on filling domain gaps (e.g., e-commerce reviews and domain-specific news).

---

## 4. Methodological Foundations & Standards

To ensure scientific validity, NepSent-Engine adopts established NLP and statistical standards:

### 4.1 Annotation Agreement Theory
1. **Unweighted Cohen's Kappa ($\kappa$)** (Cohen, 1960):
   Measures pairwise nominal agreement factoring out chance agreement:
   $$\kappa = \frac{P_o - P_e}{1 - P_e}$$
   Used for global nominal agreement.
2. **Weighted Cohen's Kappa ($\kappa_w$)** (Cohen, 1968):
   Sentiment polarity is inherently **ordinal** ($\text{Negative} < \text{Neutral} < \text{Positive}$). Disagreeing between Negative and Positive is a severe error (distance = 2), whereas disagreeing between Neutral and Positive is a milder borderline disagreement (distance = 1).
   Linear weighting matrix:
   $$w_{ij} = 1 - \frac{|i - j|}{k - 1}$$
   NepSent-Engine will report both unweighted and linear weighted $\kappa$.
3. **Krippendorff's Alpha ($\alpha$)** (Krippendorff, 2004):
   Applicable across variable numbers of annotators and handles missing data. Minimum reliability threshold: $\alpha \ge 0.667$ for tentative conclusions, $\alpha \ge 0.800$ for robust benchmarks.
4. **Artstein & Poesio (2008) Protocol**:
   Standardized survey on agreement in computational linguistics: requires explicit pilot calibration, documented confusion matrices, and separate reporting of individual category bias ($P_e$).

### 4.2 Statistical Significance & Confidence Intervals
- **Bootstrap Resampling for NLP** (Berg-Kirkpatrick et al., EMNLP 2012; Efron & Tibshirani, 1993):
  Reporting point estimates of Macro-F1 (e.g., $53.1\%$) without variance bounds is invalid on low-resource test sets.
  NepSent-Engine will compute **95% Bootstrap Confidence Intervals** via 1,000 resamples for all reported test metrics.
- **Hypothesis Testing**:
  McNemar's test (for paired model predictions) and paired permutation tests will be used before claiming that Model A (e.g., MuRIL) is statistically superior to Model B (e.g., TF-IDF).

### 4.3 Data Documentation Standards
- **Data Statements for Natural Language Processing** (Bender & Friedman, 2018):
  Every curated dataset will include a formal Data Statement documenting curation rationale, language variety, speaker demographics, speech situation, text characteristics, and annotator demographics.
- **Datasheets for Datasets** (Gebru et al., 2021):
  Structured documentation covering motivation, composition, collection process, preprocessing, uses, distribution, and maintenance.

---

## 5. Literature Search & Verification Protocol

For any future citation or baseline claim added to NepSent-Engine:

1. **Verification Requirement:**
   - Must resolve to a valid DOI, ACL Anthology URL, arXiv identifier, or peer-reviewed conference/journal publication.
   - BibTeX entry must be stored in `docs/references.bib`.
2. **Extraction Protocol:**
   - For every comparative model mentioned, extract: exact training split, exact metric definition (macro vs. micro vs. weighted F1), random seed, and hardware/runtime environment.
3. **Prohibition of Hallucinated Baselines:**
   - Under no circumstances will theoretical F1 numbers (such as "mBERT target 74–82%") be presented as published baselines without a direct literature citation. All target ranges must be cited or explicitly labeled as `PROPOSED_INTERNAL_TARGET`.

---

## 6. Actionable Next Steps for Phase 0.5

1. Initialize `docs/references.bib` with verified entries for Baral et al. (2026), Patel et al. (2026), Sitoula et al. (2025), Timilsina et al. (2021), and Rauniyar et al. (2023).
2. Download sample partitions from `Shushant/NepaliSentiment` and inspect tokenization compatibility and label schemas.
3. Incorporate virama/diacritic preservation rules into the text normalization specification based on the findings of Baral et al. (2026).

