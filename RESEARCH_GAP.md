# NepSent-Engine: Research Gap & Contribution Statement

**Document:** `RESEARCH_GAP.md`  
**Status:** Canonical Scientific Specification  
**Last Updated:** November 2026  
**Core Thesis:** *Rigorous, leak-free, multi-domain sentiment evaluation in low-resource Nepali NLP.*

---

## 1. Context and Problem Statement

Sentiment analysis in high-resource languages (English, Chinese, Spanish) benefits from massive, multi-genre benchmarks (SST-5, SemEval, Yelp, Amazon Reviews) with established inter-annotator reliability metrics, standardized train/dev/test splits, and extensive subword tokenization optimizations. 

In contrast, **Nepali NLP remains severely constrained**:
1. Most existing datasets are isolated to a single online platform (primarily Twitter/X or Reddit).
2. Existing annotation efforts frequently report unverified or single-annotator labels, or selectively discard disputed sentences without analyzing the cause of disagreement.
3. Published benchmarks often suffer from unaddressed data leakage (identical sentences appearing across splits due to news wire republication or bot re-posting).
4. Transformer evaluations in Nepali rarely report statistical variance (confidence intervals) or analyze subword tokenization fragmentation on Devanagari script.

---

## 2. Identified Research Gaps

Based on our empirical repository audit and comprehensive survey of verified literature (Baral et al., 2026; Patel et al., 2026; Sitoula et al., 2025; Timilsina et al., 2021; Rauniyar et al., 2023), we identify four distinct research gaps:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        IDENTIFIED RESEARCH GAPS                        │
├──────────────────────────┬─────────────────────────────────────────────┤
│ GAP 1: Cross-Domain      │ Single-platform datasets only; no rigorous  │
│ Generalization           │ Leave-One-Domain-Out (LODO) transfer study. │
├──────────────────────────┼─────────────────────────────────────────────┤
│ GAP 2: Consensus         │ Prior work discards up to 43% of data to    │
│ Selection Bias           │ force agreement, hiding genuine ambiguity.  │
├──────────────────────────┼─────────────────────────────────────────────┤
│ GAP 3: Methodological    │ Rampant split leakage, lack of sentence-    │
│ & Leakage Rigor          │ hash grouping, absence of confidence bounds.│
├──────────────────────────┼─────────────────────────────────────────────┤
│ GAP 4: Linguistic &      │ Over-fragmentation of Devanagari tokens and │
│ Negation Failure Modes   │ misinterpretation of Nepali negation syntax.│
└──────────────────────────┴─────────────────────────────────────────────┘
```

### Gap 1: Multi-Domain Coverage and True Cross-Domain Generalization
- **Current State:** NEPSEE (Baral et al., 2026) operates strictly on tweets from Nepali public figures. NepEMO (Sitoula et al., 2025) operates on Reddit posts. Rauniyar et al. (2023) operates on anti-establishment political tweets.
- **The Gap:** How well do models trained on formal news text transfer to informal social media or product reviews? In low-resource settings, researchers often assume cross-domain transfer is viable, but no Nepali benchmark has systematically tested **Leave-One-Domain-Out (LODO)** evaluation across News/Politics, E-Commerce Reviews, and Social Media.

### Gap 2: Overcoming Consensus Selection Bias (Preserving Ambiguity)
- **Current State:** To achieve acceptable agreement scores, prior efforts have discarded substantial portions of the collected corpus. Notably, NEPSEE collected 5,614 sentences but discarded **2,425 sentences (43.2%)** where independent annotators disagreed, publishing a final dataset of only 3,189 agreed sentences.
- **The Gap:** Discarding disagreed sentences creates an **artificially simplified test bed**. Hard linguistic phenomena—such as nuanced irony, mild dissatisfaction, rhetorical questions, and double negations—are systematically purged from the benchmark. NepSent-Engine will record both:
  1. The **Consensus Partition** (for clean comparison with existing work).
  2. The **Adjudicated Partition** (preserving annotator disagreement metadata to benchmark model robustness on inherently ambiguous text).

### Gap 3: Data Leakage and Statistical Rigor in Low-Resource Benchmarking
- **Current State:** In Nepali news and social media, exact sentences repeat hundreds of times across syndication networks and social reposts. Random train/test splitting leads to severe data contamination. Furthermore, low-resource studies routinely report single-number F1 scores on small test sets ($N < 500$) without confidence intervals, obscuring statistical noise.
- **The Gap:** NepSent-Engine establishes a leak-free splitting methodology utilizing cryptographic sentence hashes as grouping keys, paired with **95% Bootstrap Resampling Confidence Intervals** and paired significance testing.

### Gap 4: Morphological Negation and Subword Fertility in Devanagari
- **Current State:** Multilingual encoders (mBERT, MuRIL) were pretrained with tokenizers that may over-fragment Nepali words into excessive subword units. Furthermore, naive rule-based error analyzers routinely fail on Nepali's agglutinative negation system (e.g., inflecting verbs with `न-` prefixes, `छैन`, `भएन`, `हुँदैन`).
- **The Gap:** There is no systematic analysis comparing subword fertility (average tokens per word) and negation detection accuracy between monolingual models (NepBERTa) and multilingual models (MuRIL, mBERT) on Nepali sentiment classification.

---

## 3. Formal Research Contribution Statement

> **Research Contribution Thesis:**  
> NepSent-Engine provides the first rigorously deduplicated, multi-domain benchmark for Nepali sentiment analysis across news, e-commerce, and social media. It introduces a double-annotated corpus with explicit ordinal agreement quantification ($\kappa$ and $\kappa_w$), preserves both consensus and adjudicated ambiguity partitions, and establishes leak-free Leave-One-Domain-Out (LODO) transfer baselines across a 4-tier model hierarchy (TF-IDF, fastText, mBERT, MuRIL/NepBERTa) supported by 95% bootstrap confidence intervals and Devanagari subword fertility auditing.

### Key Contributions:
1. **Multi-Domain Corpus:** A balanced Devanagari corpus spanning News/Politics, E-commerce Reviews, and Social Discourse with verified provenance metadata (`REAL_COLLECTED`).
2. **Dual-Layer Annotation Benchmark:** Reporting both unweighted Cohen's $\kappa$ and linear-weighted $\kappa_w$, with a documented adjudication protocol that preserves human disagreement distributions rather than discarding them.
3. **Leakage-Free Benchmark Protocol:** Deterministic group-aware train/test/validation splitting keyed on normalized sentence hashes, preventing memorization of syndicated or templated text.
4. **Cross-Domain Generalization Study:** Empirical LODO evaluation measuring domain transfer penalties between formal journalism and informal online discourse.
5. **Devanagari Subword & Negation Failure Audit:** Linguistic error taxonomy analyzing token fragmentation (fertility rates) and morphological negation failure modes.

---

## 4. Scope Boundaries: What NepSent-Engine Is NOT

To preserve academic integrity and prevent scope creep, we explicitly define non-goals:

| What NepSent-Engine IS | What NepSent-Engine IS NOT |
|---|---|
| A scientifically verified, reproducible research benchmark. | A commercial, production-ready SaaS API. |
| A high-quality corpus of 1,000–3,000 verified sentences. | An inflated claim of 50,000 unverified scraped sentences. |
| A transparent 4-tier model comparison ladder. | A claim of state-of-the-art superiority over GPT-4 without empirical evidence. |
| A study of Devanagari and Romanized code-switching challenges. | An automated translation system or synthetic text generator. |
| An open-source, fully reproducible research package. | A proprietary, closed-source model deployment. |

---

## 5. Verification Framework: Facts vs. Assumptions vs. Targets

| Category | Item Description | Current Status / Evidence |
|---|---|---|
| **Verified Fact** | NEPSEE discarded 43% of annotated sentences due to annotator disagreement. | Published in Baral et al. (2026). |
| **Verified Fact** | The previous NepSent-Engine dataset was 100% synthetic ($N=31$ unique sentences). | Confirmed in Phase Zero Audit. |
| **Verified Fact** | Standard BERT tokenizers risk stripping Devanagari virama if normalization is misconfigured. | Published in Baral et al. (2026). |
| **Assumption** | E-commerce product reviews have higher polarity and lower neutral proportion than news headlines. | Standard across NLP literature; must be confirmed on pilot Nepali collection. |
| **Assumption** | MuRIL and NepBERTa will outperform mBERT due to dedicated Indic/Nepali vocabulary representation. | Supported by Indic NLP literature; must be verified via paired significance testing. |
| **Proposed Target** | Initial pilot collection: 50–100 genuine records. | Target milestone for Phase 1.5. |
| **Proposed Target** | Full curated dataset: 1,500–3,000 double-annotated sentences across 3 domains. | Target milestone for Phase 2.5. |
| **Proposed Target** | Inter-annotator agreement target: Linear-weighted $\kappa_w \ge 0.65$ on consensus validation split. | Target metric for Phase 2. |

