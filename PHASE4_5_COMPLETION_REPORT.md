# NepSent-Engine — Phase 4.5 Completion Report

**Date:** November 3, 2026  
**Status:** Successfully Executed & Verified (100% Pass Rate across 64/64 Tests)  
**Schema Version:** `2.0-ordinal` (`0: Negative`, `1: Neutral`, `2: Positive`)  
**Methodology Standard:** Gravitas Reliability Runtime & Empirical NLP Research Rigor  

---

## 1. Executive Summary

Phase 4.5 delivered the **Unified Command-Line Interface (`nepsent`), Honest Streamlit Demonstration Application, and Continuous Integration Workflow** for NepSent-Engine.

In accordance with strict research integrity rules:
1. **Untrained / Demo Mode Blocked:** The system prohibits instantiating untrained model heads to fabricate predictions. If no valid trained checkpoint exists on disk, both the CLI and Streamlit web application block execution, displaying a prominent warning and clear instructions on how to train a valid model checkpoint.
2. **Probability Calibration Transparency:** Uncalibrated softmax outputs from neural models are never labeled as true posterior probabilities; the UI and CLI explicitly distinguish between **Platt-calibrated probabilities** (from calibrated LinearSVC and Logistic Regression) versus **uncalibrated heuristic softmax scores**.
3. **End-to-End Pipeline Accessibility:** Researchers and practitioners can execute the entire lifecycle—from raw data ingestion, hash-grouped splitting, and inter-annotator agreement to training, evaluation with 2,000 bootstrap CIs, and inference—via clean CLI subcommands.

---

## 2. Architecture & Deliverables

### A. Unified CLI Subsystem ([`nepali_scraper/cli.py`](file:///D:/Nepsent-Engine/nepali_scraper/cli.py))
Provides 6 modular subcommands powered by standard `argparse`:
- **`nepsent ingest`**: Ingests raw records from CSV, JSONL, JSON, or TXT. Applies Unicode NFC normalization, ZWJ/ZWNJ preservation, length validation, Devanagari script ratio checks ($\ge 0.70$), SHA-256 deduplication, and Hindi contamination filtering.
- **`nepsent split`**: Executes zero-leakage hash-grouped stratified splitting into `train.jsonl`, `val.jsonl`, and `test.jsonl` with structured `split_manifest.json`.
- **`nepsent iaa`**: Computes double-blind inter-annotator agreement (Unweighted Cohen's $\kappa$, Linear/Quadratic Weighted $\kappa$, and per-domain breakdown).
- **`nepsent train`**: Trains Level 0 (`majority`), Level 1 (`tfidf_logreg`, `tfidf_svm`), or Level 2 (`fasttext`) baseline models and saves serialized checkpoints to disk.
- **`nepsent evaluate`**: Evaluates test partitions against trained checkpoints, outputting point metrics, 95% empirical percentile bootstrap confidence intervals ($B = 2,000$ resamples), and linguistic error taxonomies.
- **`nepsent predict`**:
  * Safe Checkpoint Loader: Strictly validates model weights on disk before inference.
  * Calibration Readout: Outputs `is_calibrated: True` and `"Platt-Calibrated Probability"` or `"Uncalibrated Softmax Score (Heuristic Only)"`.
  * Real-time Linguistic Readout: Reports Devanagari ratio, token count, genuine negation markers, and code-mixing indicators.

### B. Public Interactive Demonstration ([`app.py`](file:///D:/Nepsent-Engine/app.py))
- **Live Linguistic Readout**: Real-time Devanagari character count, word tokens, Devanagari ratio, morphological negation detection, and script code-mixing analysis.
- **Dynamic Checkpoint Scanner**: Scans `checkpoints/` directory for verified model weights. If empty, displays an explicit blocking callout with exact training commands.
- **Side-by-Side Model Comparison**: Enables multi-model selection to compare predictions and probability distributions simultaneously.
- **Methodological Calibration Notice**: Prominently informs evaluators about cross-domain distribution shifts, low-resource sarcasm nuances, and softmax overconfidence.

### C. Continuous Integration Workflow ([`.github/workflows/ci.yml`](file:///D:/Nepsent-Engine/.github/workflows/ci.yml))
- Automated GitHub Actions matrix across Python `3.10`, `3.11`, and `3.12` on `ubuntu-latest`.
- Installs dependencies from `requirements.txt`, validates environment imports, and executes `pytest -v`.

### D. End-to-End CLI Verification Suite ([`tests/test_cli.py`](file:///D:/Nepsent-Engine/tests/test_cli.py))
- 5 comprehensive tests validating CLI parser construction, ingest-split lifecycle, IAA computation, model training, calibrated prediction, bootstrap evaluation, and untrained checkpoint rejection.

---

## 3. Full Repository Test Results

```text
============================= test session starts =============================
platform win32 -- Python 3.15.1, pytest-9.2.0, pluggy-1.7.0 -- D:\Nepsent-Engine\venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\Nepsent-Engine
plugins: anyio-4.16.0
collected 64 items

tests/test_annotation_and_splitting.py::test_exact_deduplication_via_sha256 PASSED [  1%]
tests/test_annotation_and_splitting.py::test_near_duplicate_detection_via_jaccard PASSED [  3%]
tests/test_annotation_and_splitting.py::test_hindi_contamination_detection PASSED [  4%]
tests/test_annotation_and_splitting.py::test_devanagari_ratio_enforcement PASSED [  6%]
tests/test_annotation_and_splitting.py::test_perfect_iaa_agreement PASSED [  7%]
tests/test_annotation_and_splitting.py::test_ordinal_weighted_kappa_penalizes_extreme_mismatches_more PASSED [  9%]
tests/test_annotation_and_splitting.py::test_iaa_per_domain_breakdown PASSED [ 10%]
tests/test_annotation_and_splitting.py::test_export_for_annotation_blinds_labels PASSED [ 12%]
tests/test_annotation_and_splitting.py::test_adjudicate_annotations_workflow PASSED [ 14%]
tests/test_annotation_and_splitting.py::test_hash_grouped_stratified_split_zero_leakage PASSED [ 15%]
tests/test_annotation_and_splitting.py::test_splitter_statistical_power_warning PASSED [ 17%]
tests/test_cli.py::test_cli_parser_build PASSED                          [ 18%]
tests/test_cli.py::test_cli_ingest_and_split_pipeline PASSED             [ 20%]
tests/test_cli.py::test_cli_iaa_workflow PASSED                          [ 21%]
tests/test_cli.py::test_cli_train_predict_and_evaluate_lifecycle PASSED  [ 23%]
tests/test_cli.py::test_cli_predict_untrained_checkpoint_blocked PASSED  [ 25%]
tests/test_config.py::test_project_root_is_path PASSED                   [ 26%]
tests/test_config.py::test_directories_resolve_relative_to_root PASSED   [ 28%]
tests/test_config.py::test_no_hardcoded_drive_letters_in_paths PASSED    [ 29%]
tests/test_config.py::test_schema_version_and_ordinal_mapping PASSED     [ 31%]
tests/test_config.py::test_legacy_schema_preserved PASSED                [ 32%]
tests/test_config.py::test_import_produces_no_stdout_pollution PASSED    [ 34%]
tests/test_evaluation_and_linguistics.py::test_bootstrap_confidence_intervals_bounds PASSED [ 35%]
tests/test_evaluation_and_linguistics.py::test_paired_bootstrap_hypothesis_testing PASSED [ 37%]
tests/test_evaluation_and_linguistics.py::test_mcnemar_test_exact_and_chi2 PASSED [ 39%]
tests/test_evaluation_and_linguistics.py::test_nepali_negation_detector_flags_authentic_negation PASSED [ 40%]
tests/test_evaluation_and_linguistics.py::test_nepali_negation_detector_does_not_false_fire_on_nouns_with_na PASSED [ 42%]
tests/test_evaluation_and_linguistics.py::test_code_mixing_analyzer PASSED [ 43%]
tests/test_evaluation_and_linguistics.py::test_subword_fertility_calculator PASSED [ 45%]
tests/test_evaluation_and_linguistics.py::test_evaluate_per_domain_and_power_warning PASSED [ 46%]
tests/test_evaluation_and_linguistics.py::test_run_lodo_evaluation_generalization PASSED [ 48%]
tests/test_evaluation_and_linguistics.py::test_error_taxonomy_categorization PASSED [ 50%]
tests/test_models.py::test_majority_baseline_classifier_round_trip PASSED [ 51%]
tests/test_models.py::test_tfidf_logistic_regression_round_trip PASSED   [ 53%]
tests/test_models.py::test_tfidf_linear_svc_calibrated_probabilities PASSED [ 54%]
tests/test_models.py::test_subword_fasttext_classifier_round_trip PASSED [ 56%]
tests/test_models.py::test_transformer_forward_pass_mock_config PASSED   [ 57%]
tests/test_models.py::test_transformer_sentiment_trainer_save_structure PASSED [ 59%]
tests/test_processors.py::test_unicode_nfc_normalization PASSED          [ 60%]
tests/test_processors.py::test_nfc_normalization_preserves_virama PASSED [ 62%]
tests/test_processors.py::test_clean_non_semantic_zero_width_spaces PASSED [ 64%]
tests/test_processors.py::test_preserve_devanagari_zwj_and_zwnj PASSED   [ 65%]
tests/test_processors.py::test_collapse_consecutive_zero_width_joiners PASSED [ 67%]
tests/test_processors.py::test_sentence_segmentation_with_purna_viram PASSED [ 68%]
tests/test_processors.py::test_sentence_segmentation_with_question_and_exclamation PASSED [ 70%]
tests/test_processors.py::test_sentence_segmentation_with_double_danda PASSED [ 71%]
tests/test_processors.py::test_sentence_segmentation_preserves_trailing_sentence_without_delimiter PASSED [ 73%]
tests/test_processors.py::test_sentence_segmentation_preserves_trailing_sentence_after_delimited_one PASSED [ 75%]
tests/test_processors.py::test_sentence_segmentation_with_repeated_delimiters PASSED [ 76%]
tests/test_processors.py::test_sentence_segmentation_with_newlines PASSED [ 78%]
tests/test_processors.py::test_inclusive_min_length_boundary PASSED      [ 79%]
tests/test_processors.py::test_below_min_length_is_excluded PASSED       [ 81%]
tests/test_processors.py::test_inclusive_max_length_boundary PASSED      [ 82%]
tests/test_processors.py::test_devanagari_ratio_pure_nepali PASSED       [ 84%]
tests/test_processors.py::test_devanagari_ratio_code_mixed PASSED        [ 85%]
tests/test_processors.py::test_devanagari_ratio_pure_english PASSED      [ 87%]
tests/test_processors.py::test_empty_string_devanagari_ratio PASSED      [ 89%]
tests/test_processors.py::test_hash_invariance_to_whitespace PASSED      [ 90%]
tests/test_processors.py::test_hash_invariance_to_unicode_decomposition PASSED [ 92%]
tests/test_processors.py::test_distinct_sentences_yield_distinct_hashes PASSED [ 93%]
tests/test_processors.py::test_processor_instance_hash_matches_standalone PASSED [ 95%]
tests/test_processors.py::test_batch_process_segmentation PASSED         [ 96%]
tests/test_processors.py::test_batch_process_normalization PASSED        [ 98%]
tests/test_processors.py::test_processor_statistics PASSED               [100%]

============================= 64 passed in 19.82s =============================
```

