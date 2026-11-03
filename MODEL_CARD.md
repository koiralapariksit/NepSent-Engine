# Model Card: NepSent-Engine 4-Tier Sentiment Model Ladder

Following the model card documentation standard proposed by **Mitchell et al. (2019)**.

---

## 1. Model Overview

NepSent-Engine structures sentiment classification across a **4-tier inductive ladder**, establishing baseline performance from non-parametric heuristics up to fine-tuned deep multilingual transformers:

```
Level 0: MajorityBaselineClassifier (Empirical training class prior)
   |
Level 1: TfidfLinearModel (LogisticRegression & Platt-Calibrated LinearSVC)
   |
Level 2: SubwordFastTextClassifier (Character n-grams 3-5 + Linear SGD)
   |
Level 3: Pretrained Multilingual Transformers (mBERT, MuRIL)
   |
Level 4: Pretrained Monolingual Nepali Transformer (NepBERTa)
```

---

## 2. Intended Use & Target Tasks

- **Primary Task:** 3-class ordinal sentiment classification for Nepali text:
  - `0`: Negative (dissatisfaction, criticism, anger, sadness)
  - `1`: Neutral (factual reporting, objective updates, balanced sentiment)
  - `2`: Positive (satisfaction, praise, joy, recommendation)
- **Domain Scope:** News headlines, e-commerce consumer reviews, public conversational discourse.
- **Out-of-Scope Uses:**
  - Automated legal, political, or employment decisions based on sentiment polarity.
  - Sarcasm or satire detection without human verification.
  - Languages written in scripts other than Devanagari.

---

## 3. Tier Specifications & Hyperparameters

### Level 0: Majority Class Baseline
- **Description:** Always predicts the empirical majority class from the training partition; outputs training class priors as probability distribution.
- **Role:** Sets the absolute empirical lower bound to reveal whether models are outperforming class imbalance.

### Level 1: TF-IDF Linear Models
- **Features:** Word n-grams `(1, 2)` combined with character boundary n-grams `(2, 4)` (`char_wb`), capturing word stems and Devanagari inflections.
- **Logistic Regression:**
  - Regularization: L2 ($C=1.0$)
  - Class Weights: `balanced` (inversely proportional to class frequencies)
  - Solver: L-BFGS, max iterations = 1000
- **LinearSVC:**
  - Regularization: L2 ($C=1.0$)
  - Probability Calibration: Platt scaling via `CalibratedClassifierCV` with dynamic fold adaptation (`cv = min(3, min_class_count)`).

### Level 2: Subword Morphological Classifier (FastText Equivalent)
- **Features:** Character n-grams `(3, 5)` addressing out-of-vocabulary agglutination and verbal suffixes without C++ compiler failures.
- **Classifier:** Calibrated SGD with log-loss, elastic-net regularization, and balanced class weights.

### Level 3 & 4: Fine-Tuned Pretrained Transformers
- **Architectures:**
  1. `bert-base-multilingual-cased` (mBERT): 104 languages, WordPiece tokenizer.
  2. `google/muril-base-cased` (MuRIL): Pretrained on 17 Indian languages and English, including Nepali text and translation pairs.
  3. `Rajan/NepBERTa`: Monolingual Nepali RoBERTa trained on 500MB+ of native Nepali web text.
- **Fine-Tuning Hyperparameters:**
  - Maximum Sequence Length: 128 tokens
  - Optimizer: AdamW ($\text{lr} = 2 \times 10^{-5}$, $\epsilon = 10^{-8}$)
  - Batch Size: 16
  - Epochs: 3–5 with early stopping on validation Macro F1
  - Device: CUDA GPU with automatic CPU fallback

---

## 4. Evaluation Methodology

In accordance with strict research invariants:
- **No Binomial Assumptions on Macro F1:** Models are evaluated via **non-parametric empirical percentile bootstrap ($B = 2,000$ resamples)**, yielding true distribution-free 95% confidence intervals.
- **Statistical Significance:** Model comparisons report empirical two-tailed $p$-values via paired bootstrap difference testing and McNemar's contingency testing.
- **Cross-Domain Generalization:** Leave-One-Domain-Out (LODO) cross-validation is performed to measure transfer from news to reviews and social media.

---

## 5. Technical Limitations & Failure Modes

1. **Uncalibrated Softmax Overconfidence:** Deep transformers outputting raw softmax values display overconfident probabilities on out-of-distribution text.
2. **Morphological Negation Inversion:** Negations embedded in verbal suffixes (`गरेन`, `जान्नँ`) are occasionally missed by word-level tokenizers.
3. **Honorific Sarcasm:** High-register polite phrasing (`हजुरको सेवा धेरै महान छ...`) conveying sarcastic contempt is frequently misclassified as Positive.

