# NepSent-Engine: Limitations & Known Failure Modes

This document details the known technical, linguistic, and domain-specific failure modes of the models and evaluation systems within NepSent-Engine.

---

## 1. Linguistic & Morphological Failure Modes

### 1.1 Sarcasm and Indirect Politeness (व्यङ्ग्य र शिष्टाचार)
Nepali social discourse often employs high-register honorifics (`हजुर`, `तपाईं`, `पाल्नुभयो`) and positive vocabulary to deliver biting sarcasm or ironic contempt:
- *Example:* "हजुरको यति राम्रो सेवा देख्दा म त छक्क परेँ, सामान चाहिँ ३ महिनापछि पनि आइपुगेन!"
- *Model Failure:* N-gram and bag-of-words models latch onto the high frequency of positive terms (`हजुर`, `राम्रो`, `छक्क परेँ`) and classify the sentence as Positive (2), completely missing the negative contempt.

### 1.2 Morphological Negation Inversion
In Nepali grammar, negation is frequently encoded directly into verbal inflectional suffixes:
- Positive verb: `गर्यो` (did) $\rightarrow$ Negative verb: `गरेन` (did not do)
- Positive verb: `जान्छु` (I go) $\rightarrow$ Negative verb: `जान्नँ` (I will not go)
- Positive verb: `हुनेछ` (will happen) $\rightarrow$ Negative verb: `हुँदैन` (will not happen / not permitted)
When models tokenize sentences into words or subwords with high fertility, the negation suffix can be fragmented or diluted across positive context words, causing polarity inversion errors.

### 1.3 Code-Mixing and Romanized Nepali (अंग्रेजी-नेपाली मिश्रित भाषा)
Conversational texts on social media and e-commerce platforms feature heavy script code-mixing:
- *Example:* "Delivery एकदमै slow भयो, totally disappointed!"
Monolingual Devanagari models often discard Latin tokens during preprocessing, while multilingual models may misinterpret Romanized Nepali transliteration (e.g. *ramro* $\ne$ English word).

---

## 2. Statistical & Evaluative Limitations

### 2.1 Domain Generalization Degradation (Cross-Domain Shift)
Models trained exclusively on structured news journalism (`news_politics`) show sharp performance drops when evaluated out-of-domain on consumer product reviews (`ecommerce_reviews`) or colloquial Twitter/Reddit posts (`social_media`). This gap is documented via our Leave-One-Domain-Out (LODO) evaluator.

### 2.2 Small-Sample Test Variance (Statistical Power Safeguard)
When evaluating domain slices with sample size $N < 50$ (or per-class support $< 80$), non-parametric bootstrap confidence interval widths expand dramatically ($\pm 10\text{--}15\%$). NepSent-Engine automatically marks such slices as `is_exploratory: True`.

### 2.3 Uncalibrated Softmax Probabilities
Softmax outputs from uncalibrated deep neural networks (mBERT, MuRIL, NepBERTa) do not represent true Bayesian posterior probabilities and tend toward overconfidence. Unless Platt scaling or temperature scaling is explicitly trained, softmax outputs should be interpreted as raw heuristic ranking scores.

