# Dataset Card: NepSent Multi-Domain Nepali Sentiment Dataset

Following the documentation standards proposed in:
- **Datasheets for Datasets** (Gebru et al., 2021)
- **Data Statements for Natural Language Processing** (Bender & Friedman, 2018)

---

## 1. Dataset Description

- **Dataset Name:** NepSent Multi-Domain Nepali Sentiment Dataset
- **Language:** Nepali (`ne` / `nep_Deva`)
- **Primary Script:** Devanagari Unicode (U+0900–U+097F) with code-mixed English/Roman terms
- **Domains Represented:**
  1. `news_politics`: Formal Nepali news journalism (Ratopati, Setopati, OnlineKhabar)
  2. `ecommerce_reviews`: Consumer product and merchant reviews (Daraz Nepal)
  3. `social_media`: Public conversational sentiment (Twitter/X Nepali discourse)
- **Annotation Schema:** Ordinal 3-class (`schema_version: 2.0-ordinal`)
  - `0`: Negative
  - `1`: Neutral
  - `2`: Positive

---

## 2. Data Curation & Preprocessing

### 2.1 Unicode Normalization
All collected texts undergo canonical Unicode Form C (NFC) normalization.
- **Conjunct Preservation:** Essential Devanagari zero-width joiners (ZWJ `\u200D` and ZWNJ `\u200C`) required for ligatures and eyelash-ra are strictly preserved.
- **Invisible Whitespace Stripping:** Non-semantic invisible characters (BOM `\uFEFF`, zero-width space `\u200B`, left-to-right/right-to-left marks `\u200E`/`\u200F`, soft hyphens `\u00AD`) are systematically purged.

### 2.2 Boundary Segmentation & Filtering
- **Sentence Segmentation:** Tokenized along Nepali punctuation delimiters (Purna Viram `।`, Deergha Viram `॥`, question marks `?`, exclamation marks `!`) without dropping trailing sentences.
- **Length Constraints:** Inclusive character boundaries: $15 \le \text{length} \le 350$. Sentences below 15 characters lack sufficient contextual sentiment; sentences exceeding 350 characters represent multi-clause paragraphs requiring document-level decomposition.
- **Devanagari Ratio:** Sentences must satisfy $\text{Devanagari characters} / \text{non-whitespace characters} \ge 0.70$. Sentences with lower ratios are excluded to prevent English/Hindi noise while permitting natural code-mixed loanwords (e.g., "delivery", "mobile").

### 2.3 De-Duplication & Leakage Prevention
- **Exact Deduplication:** Canonical text strings are hashed via deterministic SHA-256. Exact duplicates across different URLs or collection epochs are merged into a single canonical entry with aggregated provenance counts.
- **Near-Duplicate Detection:** Character 3-gram Jaccard similarity is computed across candidate sentences. Texts exhibiting Jaccard overlap $\ge 0.85$ are flagged and removed to prevent templated review spam.
- **Hindi Contamination Flagger:** Text tokens are cross-referenced against high-frequency Hindi-exclusive function words (`है`, `हैं`, `था`, `थी`, `क्योंकि`, `लेकिन`, `नहीं`, `किया`). Matches are flagged and dropped.

---

## 3. Human Annotation Process

- **Annotator Qualifications:** Native Nepali speakers fluent in Devanagari script and contemporary Nepali socio-political discourse.
- **Double-Blind Protocol:** Sentences are decoupled from metadata, stripped of source identifiers, randomized, and distributed to annotators independently.
- **Guidelines:** Annotators strictly adhere to [`docs/ANNOTATION_GUIDELINES.md`](file:///D:/Nepsent-Engine/docs/ANNOTATION_GUIDELINES.md).
- **Agreement Metrics:** Quality is audited using unweighted Cohen's $\kappa$ and linear weighted Cohen's $\kappa_w$ to ensure ordinal consistency ($0 \leftrightarrow 2$ mismatches are penalized with weight 2).
- **Adjudication:** Discordant annotator pairs are reviewed by a third senior linguist to reach final consensus.

---

## 4. Partitioning & Leakage Guarantees

- **Hash-Grouped Stratified Partitioning:** Splitting into 80% Train, 10% Validation, and 10% Test is performed using sentence-hash grouping (`nepali_scraper/splitter.py`).
- **Zero Leakage:** All instances of identical canonical sentences are constrained to the same partition, preventing train-to-test contamination.
- **Statistical Power:** Test sets enforce minimum class support $\ge 80$ samples per class to ensure reliable bootstrap confidence intervals.

---

## 5. Licensing & Ethical Considerations

- **Code vs. Data Separation:** The software pipeline is released under the **MIT License**. Text corpora collected from third-party platforms remain subject to source copyright and fair use / fair dealing research exemptions.
- **Personal Identifiable Information (PII):** Usernames, phone numbers, email addresses, and personal identifying metadata are scrubbed during ingestion.
- **Low-Resource Language Context:** The dataset is intended for academic research, linguistic benchmarking, and educational low-resource NLP experimentation.

