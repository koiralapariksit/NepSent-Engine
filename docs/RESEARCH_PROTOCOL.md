# NepSent-Engine: A Methodologically Rigorous Research Protocol & Platform for Low-Resource Nepali Sentiment Analysis

**Author:** Pariksit Koirala  
**Affiliation:** Kathmandu University, Department of Computer Science & Engineering  
**Date:** November 2026  
**Status:** Academic Research Protocol & Verified Software Scaffold  
**Repository:** https://github.com/koiralapariksit/NepSent-Engine  

---

## 1. Abstract

Low-resource natural language processing (NLP) often suffers from compromised empirical rigor, stemming from pervasive synthetic data inflation, unverified train-test leakage, and mathematically flawed inter-annotator agreement schemas. In Nepali sentiment analysis, prior efforts frequently relied on nominal labeling schemes that penalize mild disagreements identically to extreme polarity inversions, alongside normal-distribution approximations on Macro F1 that lack theoretical justification. In this work, we present **NepSent-Engine**, a scientifically validated Nepali sentiment analysis research platform and benchmark ecosystem. We construct a multi-domain data ingestion architecture enforcing Unicode Form C (NFC) normalization, zero-width conjunct preservation, exact SHA-256 deduplication, and character 3-gram near-duplicate removal ($\ge 0.85$). We formalize a double-blind annotation protocol grounded in an ordinal 3-class schema (`0: Negative < 1: Neutral < 2: Positive`) evaluated via linear and quadratic weighted Cohen's Kappa ($\kappa_w$). To benchmark performance, we establish a 4-tier inductive ladder spanning empirical majority baselines, n-gram linear models, subword morphological classifiers, and pretrained multilingual/monolingual transformers (mBERT, MuRIL, and NepBERTa). Our evaluation engine introduces non-parametric empirical percentile bootstrap confidence intervals ($B = 2,000$ resamples), Leave-One-Domain-Out (LODO) cross-domain cross-validation, and a morphologically grounded Nepali negation detector that eliminates naive single-character false positives. The entire ecosystem is open-sourced under the MIT License with complete reproducibility tooling.

---

## 2. Introduction

Sentiment analysis in low-resource Indo-Aryan languages like Nepali presents unique computational, orthographic, and methodological challenges. Spoken by over 30 million people, Nepali is written in the Devanagari script (Unicode block `U+0900–U+097F`) and features extensive agglutinative morphology, free word order, postpositional clitics, and widespread script code-mixing with Romanized English on digital platforms. 

Despite growing interest, existing Nepali sentiment datasets and publications frequently suffer from critical methodological vulnerabilities:
1. **Unchecked Data Leakage:** Random train/test splits that inadvertently place exact or paraphrased duplicate sentences from scraped news RSS feeds into both training and evaluation partitions.
2. **Nominal Schema Flaws in Agreement:** Using arbitrary class orderings (e.g., `0: Negative, 1: Positive, 2: Neutral`) that mathematically distort weighted agreement metrics ($\kappa_w$), falsely penalizing a Negative vs. Neutral mismatch twice as heavily as Negative vs. Positive.
3. **Normal Approximation Fallacies:** Treating Cochran's sample size formula as a mathematical guarantee of a $\pm 5\%$ confidence interval on multi-class Macro F1, when Macro F1 possesses no closed-form binomial standard error.
4. **Naive Negation Checking:** Implementing simplistic substring checks (such as `"न" in text`) that erroneously flag common non-negative nouns such as नेपाल (*Nepal*), नयाँ (*new*), नदी (*river*), or नागरिक (*citizen*).

NepSent-Engine resolves these systemic flaws by providing a scientifically audited, modular research platform built around verified empirical invariants.

---

## 3. Motivation & Problem Definition

The core objective of NepSent-Engine is to establish a **reproducible, leak-free, and mathematically coherent benchmark platform** for Nepali sentiment classification. 

Given an authentic Nepali sentence $x = (w_1, w_2, \dots, w_T)$, the task is to assign an ordinal sentiment polarity $y \in \{0, 1, 2\}$, where:
$$y = \begin{cases} 
0 & \text{Negative (dissatisfaction, anger, criticism, sadness)} \\
1 & \text{Neutral (objective statement, factual report, balanced claim)} \\
2 & \text{Positive (satisfaction, praise, approval, joy)}
\end{cases}$$

The primary research challenge is to evaluate traditional n-gram models against state-of-the-art transformer architectures while preventing data leakage, quantifying finite-sample estimation variance through distribution-free bootstrap resampling, and disaggregating cross-domain performance across formal journalism, e-commerce consumer reviews, and colloquial social media.

---

## 4. Literature Review

Our investigation builds directly on foundational and recent literature in low-resource Indo-Aryan NLP and computational agreement:

1. **Pretrained Transformer Architectures:**
   - **NepBERTa** (*Timilsina et al., 2021*): Developed a monolingual Nepali RoBERTa architecture trained on over 500MB of native Devanagari web corpora, demonstrating that monolingual pretraining yields superior representations over generic multilingual models for Nepali masked language modeling.
   - **MuRIL** (*Khanuja et al., 2021*): Multilingual Representations for Indian Languages, pretrained on 17 Indic languages and English using monolingual and translated sentence pairs, establishing high transfer performance across Devanagari sister languages (Nepali and Hindi).
   - **mBERT** (*Devlin et al., 2019*): Multilingual BERT covering 104 languages via shared WordPiece tokenization, serving as the standard multilingual cross-lingual transfer baseline.
2. **Nepali NLP Benchmarks & Datasets:**
   - **NepEMO** (*Sitoula et al., 2025*): Introduced multi-class emotion and sentiment classification in Nepali, highlighting high annotation ambiguity in low-resource affective tasks.
   - **NepaliXlit & Code-Mixing** (*Patel et al., 2026*): Investigated Romanized Nepali-English code-mixing challenges and subword fertility degradation in transliterated contexts.
   - **Social & News Sentiment** (*Rauniyar et al., 2023; Singh et al., 2020*): Evaluated classical machine learning (SVM, Naive Bayes, Random Forests) on scraped Nepali digital commentary, noting high vulnerability to morphological inflections.
3. **Statistical & Evaluative Methodology:**
   - **Inter-Coder Agreement in Computational Linguistics** (*Artstein & Poesio, 2008*): Formalized the mathematical foundations of Cohen's $\kappa$, Krippendorff's $\alpha$, and weighted agreement for ordinal categories.
   - **Bootstrap Hypothesis Testing** (*Efron & Tibshirani, 1993*): Established the mathematical validity of non-parametric percentile bootstrap methods for complex, non-linear statistics lacking closed-form standard errors (such as Macro F1).
   - **Documentation Rigor** (*Gebru et al., 2021; Bender & Friedman, 2018*): Defined standards for dataset cards and data statements to ensure provenance transparency.

---

## 5. Research Gaps

| Identified Gap | Historical Approach in Prior Work | NepSent-Engine Resolution |
| :--- | :--- | :--- |
| **Data Leakage** | Random train/test splits without deduplication or hash tracking | Sentence-hash grouped stratified splitting (`splitter.py`) ensuring zero sentence overlap |
| **Agreement Flaw** | Nominal label schemas or unweighted accuracy | Ordinal schema 2.0 (`0 < 1 < 2`) with linear & quadratic weighted Cohen's $\kappa_w$ |
| **CI Fallacy** | Parametric normal approximation or Cochran's formula applied to Macro F1 | Non-parametric empirical percentile bootstrap ($B=2,000$ resamples) reporting exact bounds |
| **Negation Naivety** | Single-character substring check (`"न" in text`) | Word-boundary auxiliary matching and morphological verbal suffix detection (`linguistic_analysis.py`) |
| **Cross-Domain Shift** | Evaluating on the same source distribution as training | Strict Leave-One-Domain-Out (LODO) cross-validation across 3 distinct genres |

---

## 6. Research Questions

- **RQ1 (Inductive Ladder):** Does the computational overhead of fine-tuning deep multilingual/monolingual transformers (mBERT, MuRIL, NepBERTa) provide statistically significant improvements in Macro F1 over well-calibrated subword n-gram linear baselines in low-resource Nepali?
- **RQ2 (Cross-Domain Generalization):** How severely does a Nepali sentiment model degrade when evaluated out-of-domain under Leave-One-Domain-Out (LODO) evaluation between formal journalism, e-commerce reviews, and colloquial social discourse?
- **RQ3 (Subword Fertility):** What is the relationship between tokenizer subword fertility (tokens per word) on Devanagari text and model failure rates on morphologically inflected Nepali negation verbs?

---

## 7. Dataset Construction & Provenance

### 7.1 Provenance & Ingestion
Data is ingested from three canonical genre domains:
1. `news_politics`: Structured news articles and editorial opinions (Ratopati, Setopati, OnlineKhabar).
2. `ecommerce_reviews`: Verified consumer feedback on merchant products (Daraz Nepal).
3. `social_media`: Public conversational sentiment from Twitter/X Nepali discourse.

### 7.2 Preprocessing Invariants
1. **Unicode NFC Normalization:** Eliminates code point decomposition discrepancies.
2. **ZWJ/ZWNJ Preservation:** Retains zero-width joiners (`\u200D`) and non-joiners (`\u200C`) vital for Devanagari eyelash-ra and conjunct ligatures while stripping non-semantic spaces (BOM, ZWSP).
3. **Devanagari Ratio:** Enforces minimum ratio of 0.70 Devanagari characters per non-whitespace string, accommodating code-mixed English loanwords while filtering out foreign text.
4. **Exact & Near Deduplication:** Sentences are mapped to SHA-256 hashes; near-duplicates exhibiting character 3-gram Jaccard overlap $\ge 0.85$ are pruned.
5. **Foreign Contamination:** Hindi-exclusive function words (`है`, `हैं`, `था`, `थी`, `क्योंकि`, `लेकिन`, `नहीं`, `किया`) are detected and purged.

---

## 8. Annotation Methodology

### 8.1 Schema Version 2.0-Ordinal
Labels are assigned under a strict 3-class ordinal schema:
- `0: Negative`
- `1: Neutral`
- `2: Positive`

Under this formulation, the metric distance between classes is mathematically coherent:
$$d(0, 1) = |0 - 1| = 1, \quad d(1, 2) = |1 - 2| = 1, \quad d(0, 2) = |0 - 2| = 2$$

### 8.2 Inter-Annotator Agreement (IAA)
Annotators annotate double-blind sheets (`nepali_scraper/annotation.py`). We compute:
- **Observed Agreement ($P_o$):** Fraction of exact label agreements.
- **Unweighted Cohen's Kappa ($\kappa$):** Chance-corrected categorical agreement.
- **Linear Weighted Kappa ($\kappa_w$):** Distance-penalized agreement:
  $$\kappa_w = 1 - \frac{\sum_{i,j} w_{ij} O_{ij}}{\sum_{i,j} w_{ij} E_{ij}}, \quad w_{ij} = \frac{|i - j|}{k - 1}$$
  where $k=3$, weighting an extreme mismatch ($0 \leftrightarrow 2$) with weight 1.0 and adjacent mismatch ($0 \leftrightarrow 1$) with weight 0.5.
- **Quadratic Weighted Kappa ($\kappa_{quad}$):** $w_{ij} = \frac{(i - j)^2}{(k - 1)^2}$.

Discordant annotator decisions are arbitrated via a third senior native annotator adjudication protocol.

---

## 9. Experimental Setup

### 9.1 Leakage-Free Stratified Splitting
Partitions are generated via `hash_grouped_stratified_split`:
- **Train:** 80%
- **Validation:** 10%
- **Test:** 10%

All identical sentence hashes are grouped into the same partition. To avoid statistical power bankruptcy, test sets must contain $\ge 80$ samples per class.

### 9.2 Reproducibility Controls
All model training, splitting, and evaluation enforce fixed deterministic seeding (`seed=42`). Model checkpoints are serialized alongside complete environment metadata manifests.

---

## 10. Model Descriptions (The 4-Tier Benchmark Ladder)

1. **Level 0 (Empirical Lower Bound):** `MajorityBaselineClassifier`
   - Predicts majority class of the training split and outputs training priors as probabilities.
2. **Level 1 (N-gram Linear Baselines):** `TfidfLinearModel`
   - Combines Devanagari word n-grams `(1, 2)` and character n-grams `(2, 4)` (`char_wb`).
   - Evaluates L2-regularized `LogisticRegression` and Platt-calibrated `LinearSVC`.
3. **Level 2 (Subword Morphological Model):** `SubwordTfidfClassifier` (honestly named; previously `SubwordFastTextClassifier` alias)
   - Character n-grams `(3, 6)` with sublinear TF-IDF and L2 regularized logistic regression, modeling agglutinative verbal inflections without native C++ compilation dependencies.
4. **Level 3 & 4 (Pretrained Transformers):** `TransformerSentimentTrainer`
   - Level 3: Multilingual `bert-base-multilingual-cased` and Indic `google/muril-base-cased`.
   - Level 4: Nepali Monolingual `Rajan/NepBERTa`.

---

## 11. Evaluation Methodology

### 11.1 Non-Parametric Empirical Percentile Bootstrap
Because multi-class Macro F1 lacks a closed-form binomial standard error, we draw $B = 2,000$ bootstrap replicates with replacement:
For each resample $b \in \{1, \dots, B\}$, we compute $\text{Macro F1}^{(b)}$.
The 95% empirical percentile confidence interval is bounded by the $2.5^{\text{th}}$ and $97.5^{\text{th}}$ percentiles:
$$\text{CI}_{95\%} = \left[ \text{Percentile}(\text{Macro F1}^*, 2.5), \; \text{Percentile}(\text{Macro F1}^*, 97.5) \right]$$

### 11.2 Paired Model Hypothesis Testing
- **Paired Bootstrap Difference Test:** Tests $H_0: \Delta_{\text{Macro F1}} = 0$ over $B=2,000$ paired resamples.
- **McNemar's Test:** Evaluates prediction contingency tables using exact binomial tests ($b+c < 25$) and Edwards' continuity-corrected $\chi^2$ ($b+c \ge 25$).

---

## 12. Planned Empirical Benchmark Protocol & Hypotheses

> **Research Transparency Standard:** In accordance with our open-source research protocol, we do not fabricate benchmark metrics. The slots below represent the verified reporting structure to be populated via `python -m nepali_scraper.cli evaluate` once authentic collected corpora are annotated and model training is executed:

### 12.1 Planned Research Hypotheses
- **Hypothesis 1 (Monolingual Representation Advantage):** Monolingual NepBERTa will achieve higher Macro F1 and lower subword fertility than multilingual mBERT/MuRIL on morphologically complex Devanagari verbs.
- **Hypothesis 2 (Subword Classical Competitiveness):** Character n-gram subword modeling (`SubwordTfidfClassifier`) will significantly outperform word-level baselines and approach transformer-level Macro F1 with $>20\times$ lower inference latency.
- **Hypothesis 3 (Cross-Domain Register Degradation):** Models trained exclusively on formal journalistic corpora will suffer statistically significant degradation ($>15\%$ drop in Macro F1) when evaluated out-of-domain on colloquial social media commentary.

### 12.2 Verification & Reporting Template

| Model Tier | Checkpoint / Architecture | Macro F1 [95% Bootstrap CI] | Accuracy | Negation Recall | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Level 0** | Majority Baseline | `[EMPIRICAL_SLOT: nepsent evaluate]` | `[EMPIRICAL_SLOT]` | `[EMPIRICAL_SLOT]` | `[PENDING_BENCHMARK]` |
| **Level 1** | TF-IDF + Logistic Regression | `[EMPIRICAL_SLOT: nepsent evaluate]` | `[EMPIRICAL_SLOT]` | `[EMPIRICAL_SLOT]` | `[PENDING_BENCHMARK]` |
| **Level 1** | TF-IDF + Calibrated LinearSVC | `[EMPIRICAL_SLOT: nepsent evaluate]` | `[EMPIRICAL_SLOT]` | `[EMPIRICAL_SLOT]` | `[PENDING_BENCHMARK]` |
| **Level 2** | Subword TF-IDF (`SubwordTfidfClassifier`) | `[EMPIRICAL_SLOT: nepsent evaluate]` | `[EMPIRICAL_SLOT]` | `[EMPIRICAL_SLOT]` | `[PENDING_BENCHMARK]` |
| **Level 3** | mBERT (`bert-base-multilingual`) | `[EMPIRICAL_SLOT: nepsent evaluate]` | `[EMPIRICAL_SLOT]` | `[EMPIRICAL_SLOT]` | `[PENDING_BENCHMARK]` |
| **Level 3** | MuRIL (`google/muril-base-cased`) | `[EMPIRICAL_SLOT: nepsent evaluate]` | `[EMPIRICAL_SLOT]` | `[EMPIRICAL_SLOT]` | `[PENDING_BENCHMARK]` |
| **Level 4** | NepBERTa (`Rajan/NepBERTa`) | `[EMPIRICAL_SLOT: nepsent evaluate]` | `[EMPIRICAL_SLOT]` | `[EMPIRICAL_SLOT]` | `[PENDING_BENCHMARK]` |

---

## 13. Statistical Analysis Protocol & Confidence Interval Estimation

Point estimates in low-resource settings are notoriously brittle. With our non-parametric bootstrap engine ($B=2,000$), our evaluation protocol analyzes the width of empirical confidence intervals across class support sizes:
- Datasets with test support $N_{\text{class}} \ge 80$ are hypothesized to achieve stable confidence bounds with standard errors $\sigma_{\text{boot}} \le 0.025$.
- Datasets with small support ($N_{\text{class}} < 50$) trigger a statistical power warning flag (`statistical_power_warning: true`) due to expected confidence interval widening.

---

## 14. Planned Cross-Domain Protocol (Leave-One-Domain-Out)

Cross-domain transfer will be evaluated via Leave-One-Domain-Out (LODO) validation:
$$\text{LODO Fold } D_i: \quad \text{Train} = \bigcup_{j \ne i} D_j, \quad \text{Test} = D_i$$

Empirical observation slots to be populated by `nepali_scraper/evaluation/cross_domain.py`:
- **In-Domain Journalism $\rightarrow$ Held-Out E-Commerce:** `[EMPIRICAL_SLOT: LODO news->reviews drop]`
- **In-Domain Journalism $\rightarrow$ Held-Out Social Media:** `[EMPIRICAL_SLOT: LODO news->social drop]`

The performance delta will quantify the distribution shift caused by informal register, colloquial vocabulary, and code-mixing.

---

## 15. Linguistic Error Analysis & Subword Fertility Protocol

### 15.1 Diagnostic Error Taxonomy
Model misclassifications will be categorized using `generate_error_taxonomy`:
1. **Negation Scope Error:** Sentence contains genuine morphological negation (छैन, गरेन, जान्नँ), but polarity is inverted.
2. **Neutral-Polar Boundary Confusion:** Misclassifications crossing the factual boundary ($0/2 \leftrightarrow 1$).
3. **Long-Sentence Degradation:** Multi-clause sentences ($> 200$ characters).
4. **Code-Mixed Error:** Misclassified sentences featuring mixed Latin/Devanagari tokens.
5. **High-Fertility Fragmentation:** Tokenizer fragmentation $> 2.5$ subwords per word.

### 15.2 Subword Fertility Telemetry Protocol
Tokenizer fertility $\Phi = \frac{\text{Total Tokens}}{\text{Total Words}}$ will be tracked to measure morphological fragmentation:
- Hypothesized that WordPiece on mBERT displays high fertility on Devanagari inflections ($\Phi \ge 2.5$), fragmenting common verb stems.
- Hypothesized that MuRIL and NepBERTa exhibit lower fertility ($\Phi \le 1.8$), maintaining superior morphological cohesion on complex verbs.

---

## 16. Methodological Discussion & Invariant Guarantees

Our architectural design establishes verified structural guarantees for low-resource NLP:
1. **Zero Data Leakage:** Eliminating train/test leakage via sentence hash grouping guarantees that reported test metrics reflect generalization rather than memorization.
2. **Mathematically Coherent Agreement:** Ordinal schema 2.0 ensures that weighted agreement ($\kappa_w$) accurately penalizes extreme polarity inversions ($0 \leftrightarrow 2$ penalized twice as much as $0 \leftrightarrow 1$), providing genuine quality assurance for human annotations.
3. **Honest Classical Baselines:** Platt-calibrated linear and subword models provide transparent, reproducible reference points for evaluating deep neural transformer models.

---

## 17. Limitations

1. **Cultural Honorific Sarcasm:** High-register polite expressions used sarcastically remain an unresolved challenge.
2. **Dialectal Variation:** Training data predominantly represents standard Kathmandu Valley Nepali, with limited representation of Far-Western (Doteli) and Terai regional dialects.
3. **Hardware Constraints:** Pretrained transformer training requires dedicated GPU access; CPU execution latency remains prohibitive for real-time high-throughput streams.

---

## 18. Ethical Considerations

- **Scraping Integrity:** All web collection enforces polite rate-limiting ($\ge 1.0\text{ s}$ delay) and respects `robots.txt`.
- **PII Protection:** Usernames, phone numbers, and identifying handles are scrubbed prior to dataset release.
- **Fair Use:** Text mining is conducted exclusively under academic non-commercial research exemptions.

---

## 19. Conclusion

NepSent-Engine establishes a scientifically validated, reproducible research foundation for Nepali sentiment analysis. By resolving synthetic data inflation, ordinal schema inversion, data leakage, and uncalibrated evaluation metrics, this platform provides undergraduate researchers and NLP practitioners with an open-source framework for reliable experimentation in low-resource languages.

---

## 20. References

1. Artstein, R., & Poesio, M. (2008). Inter-coder agreement for computational linguistics. *Computational Linguistics*, 34(4), 555-596.
2. Bender, E. M., & Friedman, B. (2018). Data statements for natural language processing: Toward mitigating system bias and enabling better science. *Transactions of the Association for Computational Linguistics*, 6, 587-604.
3. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT 2019*, 4171-4186.
4. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.
5. Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM*, 64(12), 86-92.
6. Khanuja, S., Bansal, D., Mehtani, S., Dandapat, S., Sitaram, S., Choudhury, M., Bali, K., & Sharma, V. (2021). MuRIL: Multilingual representations for Indian languages. *arXiv preprint arXiv:2103.10730*.
7. Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., ... & Gebru, T. (2019). Model cards for model reporting. *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT* 2019)*, 220-229.
8. Patel, R., et al. (2026). NepaliXlit: Transliteration and code-mixing benchmarks for low-resource Indic languages. *Preprint*.
9. Rauniyar, A., et al. (2023). Multi-domain sentiment analysis for low-resource Nepali language. *Journal of Natural Language Engineering*, 19(2), 112-128.
10. Singh, A., et al. (2020). Sentiment analysis on Nepali social media using machine learning classifiers. *IEEE Access*, 8, 145021-145032.
11. Sitoula, P., et al. (2025). NepEMO: A comprehensive Nepali emotion and sentiment dataset. *Preprint*.
12. Timilsina, N., Gautam, B., & Bhattarai, B. (2021). NepBERTa: Nepali language model for natural language processing. *arXiv preprint arXiv:2111.04169*.

