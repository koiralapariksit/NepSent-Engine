# AI-Assisted Research Engineering Case Study & Educational Portfolio

**Project:** NepSent-Engine  
**Author:** Pariksit Koirala  
**Target Audience:** NLP Researchers, Research Software Engineers, and Kathmandu University Students  
**Date:** November 2026  

---

## 1. Executive Case Study: Human-in-the-Loop AI Engineering

The transformation of **NepSent-Engine** from an initial prototype into an open-source, scientifically validated research platform provides a transparent case study in pairing autonomous AI coding agents with rigorous human scientific oversight.

### 1.1 Where AI Coding Agents Excelled
1. **Rapid Codebase Exploration & Architectural Auditing:**
   - Scanning hundreds of lines of legacy Python scripts to map data dependencies, hardcoded paths, and circular imports.
2. **Devanagari Unicode & Regex Synthesis:**
   - Constructing precise Unicode code-point regular expressions distinguishing between independent vowels (`ए`), dependent matras (`े`), and conjunct joiners (`ZWJ \u200D` / `ZWNJ \u200C`).
3. **Exhaustive Test-Driven Development (TDD):**
   - Generating 64 comprehensive, edge-case unit tests covering boundary lengths, zero-width characters, bootstrap percentiles, and CLI subprocess execution.
4. **Reproducible Pipeline Orchestration:**
   - Structuring modular packages (`collectors`, `processors`, `models`, `evaluation`, `cli`) with standard entry points and clean CI configurations.

---

## 2. Critical Human Scientific Oversight Interventions

AI coding models, if left unconstrained, exhibit specific biases toward producing plausible-looking but scientifically invalid artifacts. In NepSent-Engine, human scientific oversight intervened in four critical moments:

```
+-----------------------------------------------------------------------------------------------------+
|                              HUMAN SCIENTIFIC OVERSIGHT INTERVENTIONS                               |
+-----------------------------------------------------------------------------------------------------+
|                                                                                                     |
|  [ INTERVENTION 1: Synthetic Data Rejection ]                                                       |
|  Issue: AI proposed synthetic heuristics to populate "annotations" and simulate agreement scores.   |
|  Human Action: Blocked simulation. Mandated genuine double-blind native human annotation protocol   |
|                and strict separation between unannotated data and adjudicated gold splits.          |
|                                                                                                     |
|  [ INTERVENTION 2: Ordinal Schema Inversion Fix ]                                                   |
|  Issue: AI retained legacy nominal mapping (0: Neg, 1: Pos, 2: Neu), breaking distance metric      |
|         assumptions in weighted Cohen's Kappa (|0 - 1| was 1, but |0 - 2| was 2).                   |
|  Human Action: Enforced Schema 2.0-Ordinal (0: Negative < 1: Neutral < 2: Positive), ensuring that  |
|                extreme polarity inversions are penalized twice as heavily as neutral boundaries.    |
|                                                                                                     |
|  [ INTERVENTION 3: Non-Parametric Bootstrap for Macro F1 ]                                         |
|  Issue: AI relied on Cochran's sample size formula to claim a +/- 5% error bound on Macro F1.      |
|  Human Action: Corrected statistical fallacy: Cochran bounds single binomial proportions           |
|                (Accuracy), NOT multi-class Macro F1. Implemented empirical percentile bootstrap     |
|                (B = 2,000 resamples) with a minimum test support threshold (>= 80 per class).       |
|                                                                                                     |
|  [ INTERVENTION 4: Morphological Negation Precision ]                                              |
|  Issue: AI implemented naive substring check `"न" in text`, false-firing on common nouns like      |
|         नेपाल (Nepal), नयाँ (new), नदी (river), नागरिक (citizen).                                   |
|  Human Action: Mandated linguistic token-boundary checks, auxiliary verb tables, and Devanagari     |
|                inflectional verbal suffix regex patterns (-[े]न, -[े]नन्, -न्न, -दिन).              |
|                                                                                                     |
+-----------------------------------------------------------------------------------------------------+
```

---

## 3. Kathmandu University 60-Minute Student Workshop Blueprint

**Workshop Title:** *Building Leakage-Free Low-Resource NLP Pipelines with AI Agents*  
**Duration:** 60 Minutes  
**Level:** Undergraduate Computer Science / AI Engineering Students  
**Prerequisites:** Basic Python, Git, and foundational ML concepts.

### Workshop Timeline & Objectives

| Time | Module | Core Learning Outcomes & Hands-On Exercises |
| :---: | :--- | :--- |
| **00–10 min** | **The Low-Resource Reality** | Discuss why low-resource NLP cannot rely on massive synthetic data. Explore Devanagari Unicode encoding pitfalls (NFC vs. NFD, ZWJ/ZWNJ). |
| **10–25 min** | **Leakage Prevention & Hash Splitting** | *Exercise:* Students inspect a scraped dataset, detect near-duplicate sentences using character 3-gram Jaccard similarity, and run `nepsent split` to achieve zero-leakage hash grouping. |
| **25–40 min** | **Human Annotation & Ordinal Kappa** | *Exercise:* Students annotate 10 sentences under double-blind sheets, compute unweighted vs. linear weighted Cohen's $\kappa_w$, and observe how nominal vs. ordinal schemas alter agreement scores. |
| **40–50 min** | **The Inductive Baseline Ladder** | *Exercise:* Train a Level 1 Platt-calibrated TF-IDF LinearSVC via `nepsent train`. Compare latency, memory footprint, and calibration status against deep transformers. |
| **50–60 min** | **Bootstrap Confidence Intervals & Error Analysis** | *Exercise:* Execute `nepsent evaluate` with $B = 2,000$ bootstrap resamples. Analyze the linguistic error taxonomy breakdown (negation scope vs. code-mixing). |

### Take-Home Checklist for Students
1. Always compute sentence hashes before splitting datasets.
2. Never evaluate Macro F1 using a single point estimate without bootstrap confidence intervals.
3. Verify that your Devanagari normalizer preserves conjunct joiners (`\u200D` and `\u200C`).
4. Avoid naive substring checks for language-specific grammar.

