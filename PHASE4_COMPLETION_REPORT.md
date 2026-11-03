# NepSent-Engine — Phase 4 Completion Report

**Date:** November 3, 2026  
**Status:** Successfully Executed & Verified (100% Pass Rate across 59/59 Tests)  
**Schema Version:** `2.0-ordinal` (`0: Negative`, `1: Neutral`, `2: Positive`)  
**Methodology Standard:** Gravitas Reliability Runtime & Empirical NLP Research Rigor  

---

## 1. Executive Summary

Phase 4 implemented the **Statistical Evaluation, Cross-Domain Generalization, Subword Fertility, and Nepali Linguistic Error Taxonomy** subsystem for NepSent-Engine.

In accordance with strict research and statistical invariants:
1. **Zero Normal Approximation Fallacies:** Replaced closed-form binomial assumptions with **non-parametric empirical percentile bootstrap confidence intervals ($B = 2,000$ resamples)** for Macro F1, Accuracy, and per-class metrics.
2. **Statistically Valid Model Comparisons:** Integrated paired bootstrap hypothesis testing and McNemar's test (with exact binomial and Edwards' continuity-corrected $\chi^2$).
3. **Generalization Safeguards:** Constructed Leave-One-Domain-Out (LODO) cross-validation and per-domain slice reporting with automatic statistical power alerts for small sample slices ($N < 50$).
4. **Morphologically Grounded Linguistic Negation:** Solved the pervasive false-positive flaw of naive single-character matching (`"न" in sentence`) by implementing word-boundary token matching, auxiliary lookup, and verbal suffix / prefix morphology.
5. **Subword Fertility Telemetry:** Quantified morphological fragmentation (tokens per word, character-to-token ratio, and `[UNK]` token frequency).
6. **Diagnostic Error Taxonomy:** Categorized misclassifications into 6 actionable diagnostic error buckets.

---

## 2. Architecture & Implementation Deliverables

### A. Statistical Evaluation & Bootstrap Engine ([`nepali_scraper/evaluation/metrics.py`](file:///D:/Nepsent-Engine/nepali_scraper/evaluation/metrics.py))
- **`compute_classification_metrics(y_true, y_pred, labels=[0, 1, 2])`**:
  - Point estimates for Accuracy, Macro-F1, Weighted-F1, Micro-F1, and per-class Precision/Recall/F1 with zero division guards.
  - $3 \times 3$ confusion matrix and empirical support per class.
- **`bootstrap_confidence_intervals(y_true, y_pred, n_resamples=2000, ci=0.95, seed=42)`**:
  - Distribution-free empirical percentile bootstrap ($B=2,000$ draws with replacement).
  - Reports `point`, `ci_lower`, `ci_upper`, and empirical `std_error`.
  - Mathematically guarantees valid bounds ($\text{lower} \le \text{point} \le \text{upper}$) clamped strictly within $[0.0, 1.0]$.
- **`paired_bootstrap_test(y_true, pred_model_a, pred_model_b, n_resamples=2000, seed=42, metric="macro_f1")`**:
  - Two-tailed empirical $p$-value under $H_0: \Delta = 0$ for comparing candidate architectures.
- **`mcnemar_test(y_true, pred_model_a, pred_model_b)`**:
  - Contingency analysis of discordant predictions ($b, c$).
  - Evaluates exact binomial test for small samples ($b + c < 25$) and continuity-corrected $\chi^2$ ($\frac{(|b - c| - 1)^2}{b + c}$) for large samples ($b + c \ge 25$).

### B. Cross-Domain & Leave-One-Domain-Out Evaluator ([`nepali_scraper/evaluation/cross_domain.py`](file:///D:/Nepsent-Engine/nepali_scraper/evaluation/cross_domain.py))
- **`evaluate_per_domain(y_true, y_pred, domains)`**:
  - Disaggregates overall metrics into domain slices (`news_politics`, `ecommerce_reviews`, `social_media`).
  - Automatically flags slices with $N < 50$ as `is_exploratory: True` with high-variance warnings.
- **`run_lodo_evaluation(records, model_factory)`**:
  - Trains models on $D - 1$ domains and evaluates on the held-out domain.
  - Reports macro-averaged cross-domain Macro F1 and Accuracy across all folds.

### C. Nepali Linguistic & Error Taxonomy Analyzer ([`nepali_scraper/evaluation/linguistic_analysis.py`](file:///D:/Nepsent-Engine/nepali_scraper/evaluation/linguistic_analysis.py))
- **Morphological Negation Detector (`detect_nepali_negation(text)`)**:
  - Solves the critical research flaw where naive checks like `"न" in text` false-fire on words starting with 'न'.
  - Validated against:
    * Standalone auxiliaries/copulas: `छैन`, `होइन`, `थिएन`, `हुँदैन`, `हुन्न`, `गर्दैन`, `सक्दैन`, `मिल्दैन`, `चाहिँदैन`, `पर्दैन`, `भएन`, `लाग्दैन`.
    * Dependent matra verbal endings (`-[े]न`, `-[े]नन्`, `-[े]नौ`, e.g. `गरेन`, `लेखेन`, `बसेन`, `सुनेन`) and independent vowel endings (`-एन`, `-एनन्`, e.g. `भएन`, `आएन`).
    * First-person verbal negation: `-न्न`, `-न्नँ`, `-दिन`, `-दिनँ` (e.g. `जान्नँ`, `गर्दिन`).
    * Prefixed verbs: `नगर्नु`, `नगर्नुहोला`, `नजाउ`, `नजानुहोला`, `नकिन्नुहोला`.
    * Exception whitelist: `नेपाल`, `नयाँ`, `नदी`, `नागरिक`, `नम्र`, `नेता`, `नियम`, `विद्वान`, `भगवान`.
- **Code-Mixing & Script Analyzer (`analyze_code_mixing(text)`)**:
  - Character ratio analysis (`devanagari_ratio`, `latin_ratio`) and Latin token extraction.
- **Subword Fertility Calculator (`compute_subword_fertility(texts, tokenizer)`)**:
  - Tracks average subword tokens per word (fertility), character-to-token ratio, and `[UNK]` token frequency.
- **Diagnostic Error Taxonomy (`generate_error_taxonomy(records, y_true, y_pred, tokenizer)`)**:
  - Buckets errors into:
    1. *Negation Scope Error*
    2. *Neutral-Polar Boundary Confusion*
    3. *Long-Sentence Degradation* ($> 200$ chars / $> 35$ words)
    4. *Code-Mixed Error*
    5. *High-Fertility Fragmentation* (fertility $> 2.5$ or `[UNK]` presence)
    6. *Other Ambiguity*

---

## 3. Full Test Suite Verification

Execution of the entire repository test suite:

```text
============================= test session starts =============================
platform win32 -- Python 3.15.1, pytest-9.2.0, pluggy-1.7.0 -- D:\Nepsent-Engine\venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\Nepsent-Engine
plugins: anyio-4.16.0
collected 59 items

tests/test_annotation_and_splitting.py::test_exact_deduplication_via_sha256 PASSED [  1%]
tests/test_annotation_and_splitting.py::test_near_duplicate_detection_via_jaccard PASSED [  3%]
tests/test_annotation_and_splitting.py::test_hindi_contamination_detection PASSED [  5%]
tests/test_annotation_and_splitting.py::test_devanagari_ratio_enforcement PASSED [  6%]
tests/test_annotation_and_splitting.py::test_perfect_iaa_agreement PASSED [  8%]
tests/test_annotation_and_splitting.py::test_ordinal_weighted_kappa_penalizes_extreme_mismatches_more PASSED [ 10%]
tests/test_annotation_and_splitting.py::test_iaa_per_domain_breakdown PASSED [ 11%]
tests/test_annotation_and_splitting.py::test_export_for_annotation_blinds_labels PASSED [ 13%]
tests/test_annotation_and_splitting.py::test_adjudicate_annotations_workflow PASSED [ 15%]
tests/test_annotation_and_splitting.py::test_hash_grouped_stratified_split_zero_leakage PASSED [ 16%]
tests/test_annotation_and_splitting.py::test_splitter_statistical_power_warning PASSED [ 18%]
tests/test_config.py::test_project_root_is_path PASSED                   [ 20%]
tests/test_config.py::test_directories_resolve_relative_to_root PASSED   [ 22%]
tests/test_config.py::test_no_hardcoded_drive_letters_in_paths PASSED    [ 23%]
tests/test_config.py::test_schema_version_and_ordinal_mapping PASSED     [ 25%]
tests/test_config.py::test_legacy_schema_preserved PASSED                [ 27%]
tests/test_config.py::test_import_produces_no_stdout_pollution PASSED    [ 28%]
tests/test_evaluation_and_linguistics.py::test_bootstrap_confidence_intervals_bounds PASSED [ 30%]
tests/test_evaluation_and_linguistics.py::test_paired_bootstrap_hypothesis_testing PASSED [ 32%]
tests/test_evaluation_and_linguistics.py::test_mcnemar_test_exact_and_chi2 PASSED [ 33%]
tests/test_evaluation_and_linguistics.py::test_nepali_negation_detector_flags_authentic_negation PASSED [ 35%]
tests/test_evaluation_and_linguistics.py::test_nepali_negation_detector_does_not_false_fire_on_nouns_with_na PASSED [ 37%]
tests/test_evaluation_and_linguistics.py::test_code_mixing_analyzer PASSED [ 38%]
tests/test_evaluation_and_linguistics.py::test_subword_fertility_calculator PASSED [ 40%]
tests/test_evaluation_and_linguistics.py::test_evaluate_per_domain_and_power_warning PASSED [ 42%]
tests/test_evaluation_and_linguistics.py::test_run_lodo_evaluation_generalization PASSED [ 44%]
tests/test_evaluation_and_linguistics.py::test_error_taxonomy_categorization PASSED [ 45%]
tests/test_models.py::test_majority_baseline_classifier_round_trip PASSED [ 47%]
tests/test_models.py::test_tfidf_logistic_regression_round_trip PASSED   [ 49%]
tests/test_models.py::test_tfidf_linear_svc_calibrated_probabilities PASSED [ 50%]
tests/test_models.py::test_subword_fasttext_classifier_round_trip PASSED [ 52%]
tests/test_models.py::test_transformer_forward_pass_mock_config PASSED   [ 54%]
tests/test_models.py::test_transformer_sentiment_trainer_save_structure PASSED [ 55%]
tests/test_processors.py::test_unicode_nfc_normalization PASSED          [ 57%]
tests/test_processors.py::test_nfc_normalization_preserves_virama PASSED [ 59%]
tests/test_processors.py::test_clean_non_semantic_zero_width_spaces PASSED [ 61%]
tests/test_processors.py::test_preserve_devanagari_zwj_and_zwnj PASSED   [ 62%]
tests/test_processors.py::test_collapse_consecutive_zero_width_joiners PASSED [ 64%]
tests/test_sentence_segmentation_with_purna_viram PASSED [ 66%]
tests/test_processors.py::test_sentence_segmentation_with_question_and_exclamation PASSED [ 67%]
tests/test_processors.py::test_sentence_segmentation_with_double_danda PASSED [ 69%]
tests/test_processors.py::test_sentence_segmentation_preserves_trailing_sentence_without_delimiter PASSED [ 71%]
tests/test_processors.py::test_sentence_segmentation_preserves_trailing_sentence_after_delimited_one PASSED [ 72%]
tests/test_processors.py::test_sentence_segmentation_with_repeated_delimiters PASSED [ 74%]
tests/test_processors.py::test_sentence_segmentation_with_newlines PASSED [ 76%]
tests/test_processors.py::test_inclusive_min_length_boundary PASSED      [ 77%]
tests/test_processors.py::test_below_min_length_is_excluded PASSED       [ 79%]
tests/test_processors.py::test_inclusive_max_length_boundary PASSED      [ 81%]
tests/test_processors.py::test_devanagari_ratio_pure_nepali PASSED       [ 83%]
tests/test_processors.py::test_devanagari_ratio_code_mixed PASSED        [ 84%]
tests/test_processors.py::test_devanagari_ratio_pure_english PASSED      [ 86%]
tests/test_processors.py::test_empty_string_devanagari_ratio PASSED      [ 88%]
tests/test_processors.py::test_hash_invariance_to_whitespace PASSED      [ 89%]
tests/test_processors.py::test_hash_invariance_to_unicode_decomposition PASSED [ 91%]
tests/test_processors.py::test_distinct_sentences_yield_distinct_hashes PASSED [ 93%]
tests/test_processors.py::test_processor_instance_hash_matches_standalone PASSED [ 94%]
tests/test_processors.py::test_batch_process_segmentation PASSED         [ 96%]
tests/test_processors.py::test_batch_process_normalization PASSED        [ 98%]
tests/test_processors.py::test_processor_statistics PASSED               [100%]

======================= 59 passed in 226.89s (0:03:46) ========================
```

