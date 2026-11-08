# NepSent-Engine: Phase 1 Completion Report
## Data Engineering & Validation Foundations

**Document:** `PHASE1_COMPLETION_REPORT.md`  
**Phase:** Phase 1 (Data Engineering & Validation Foundations)  
**Date:** November 3, 2026  
**Auditor:** Senior NLP Research Engineer & ML Systems Architect  
**Status:** Completed & Verified — 100% Test Pass Rate (32/32 tests passing)  
**Next Gate:** Awaiting Approval to Begin Phase 1.5 (Pilot Collection & Dataset Validation)

---

## 1. Executive Summary & Context Corrections

Phase 1 established the core engineering and linguistic processing foundation for NepSent-Engine. All hardcoded filesystem paths were replaced with dynamic resolution, import-time stdout pollution was eliminated, and a robust Devanagari text processing engine was implemented with zero-width character handling, boundary-preserving sentence segmentation, Devanagari character ratio validation, and deterministic SHA-256 sentence hashing for leak-free deduplication.

### 1.1 Mandatory Statistical Correction Recorded
As instructed, the following critical correction has been permanently registered in the project's working context:

> **Statistical Correction (Sample Size & Macro F1):**  
> Cochran's formula bounds only a single binomial proportion (accuracy), **not** Macro F1 across 3 classes. Macro F1 has no closed-form binomial standard error and depends heavily on per-class support and variance. Therefore, $N_{\text{test}} = 384$ is **not** treated as a mathematical guarantee of a $\pm 5\%$ CI on Macro F1.  
> Instead, Phase 3 evaluation will strictly report non-parametric bootstrap confidence intervals ($B = 2,000$ resamples, 95% BCa/percentile) and enforce a minimum per-class test support of **$\ge 80\text{--}100$ verified samples**.

### 1.2 Schema Version Decision Implemented
- **Active Schema (`schema_version: "2.0-ordinal"`):**
  - `0: Negative`
  - `1: Neutral`
  - `2: Positive`
  - Enables mathematically sound ordinal weighted Kappa ($\kappa_w$) where distance between Negative and Positive is $|0 - 2| = 2$.
- **Legacy Schema (`schema_version: "1.0-nominal"`):**
  - `0: Negative, 1: Positive, 2: Neutral`
  - Retained as read-only metadata reference to ensure historical synthetic files in `data/` remain uncorrupted.

---

## 2. Exact Files Modified & Created

### 2.1 Modified Files

| File | Nature of Changes | Verification Status |
|---|---|---|
| [`nepali_scraper/config.py`](file:///d:/Nepsent-Engine/nepali_scraper/config.py) | • Dynamic path resolution: `PROJECT_ROOT = Path(__file__).resolve().parents[1]`.<br>• Zero hardcoded absolute paths (`d:/...` removed).<br>• Eliminated import-time `print()` statements.<br>• Added `SCHEMA_VERSION = "2.0-ordinal"` with `LABEL_SCHEMA` mapping `0: Neg, 1: Neu, 2: Pos`.<br>• Retained `LEGACY_LABEL_SCHEMA` for backwards compatibility.<br>• Configured empirical sample targets ($N_{\text{test}} \ge 80\text{--}100$ per class). | Verified via `tests/test_config.py` (6/6 passed) |
| [`nepali_scraper/processors.py`](file:///d:/Nepsent-Engine/nepali_scraper/processors.py) | • Implemented Unicode Form C (NFC) normalization.<br>• Implemented `clean_zero_width_characters()` to strip non-semantic ZWSP (`\u200B`), BOM (`\uFEFF`), LRM/RLM (`\u200E`, `\u200F`), while preserving ZWJ (`\u200D`) and ZWNJ (`\u200C`) for Devanagari conjuncts and eyelash-ra.<br>• Implemented boundary-preserving segmenter supporting Purna Viram (`।`), double danda (`॥`), `?`, `!`, and newlines without dropping trailing sentences.<br>• Fixed length filter to inclusive boundary: `min_length <= len(s) <= max_length`.<br>• Implemented `devanagari_ratio()` and `is_valid_sentence()` to support code-mixing validation.<br>• Implemented `compute_sentence_hash()` producing deterministic SHA-256 hex strings on normalized text. | Verified via `tests/test_processors.py` (26/26 passed) |
| [`nepali_scraper/__init__.py`](file:///d:/Nepsent-Engine/nepali_scraper/__init__.py) | • Bumped version to `2.0.0`.<br>• Exported new core processor functions (`NepaliTextProcessor`, `compute_sentence_hash`, `clean_zero_width_characters`, `devanagari_ratio`, `is_valid_nepali_text`). | Verified clean module imports |

### 2.2 Newly Created Files

| File | Nature of Content | Test Count |
|---|---|---|
| [`tests/test_config.py`](file:///d:/Nepsent-Engine/tests/test_config.py) | Unit tests verifying dynamic path resolution, absence of drive-letter hardcoding, schema 2.0-ordinal mapping, legacy schema preservation, and zero stdout pollution on import. | 6 Tests |
| [`tests/test_processors.py`](file:///d:/Nepsent-Engine/tests/test_processors.py) | Unit tests covering Unicode NFC normalization, zero-width character stripping, ZWJ preservation, sentence segmentation edge cases (trailing sentences without delimiters, repeated punctuation `??`, newlines), inclusive boundary lengths (15 and 350 chars), Devanagari character ratios, SHA-256 hash invariance, and batch processing statistics. | 26 Tests |

---

## 3. Test Suite Execution & Exact Output

The test suite was executed using `pytest` in the project's virtual environment:

```powershell
.\venv\Scripts\python.exe -m pytest -v
```

### Exact Pytest Console Output:

```text
============================= test session starts =============================
platform win32 -- Python 3.15.1, pytest-9.2.0, pluggy-1.7.0 -- D:\Nepsent-Engine\venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\Nepsent-Engine
collecting ... collected 32 items

tests/test_config.py::test_project_root_is_path PASSED                   [  3%]
tests/test_config.py::test_directories_resolve_relative_to_root PASSED   [  6%]
tests/test_config.py::test_no_hardcoded_drive_letters_in_paths PASSED    [  9%]
tests/test_config.py::test_schema_version_and_ordinal_mapping PASSED     [ 12%]
tests/test_config.py::test_legacy_schema_preserved PASSED                [ 15%]
tests/test_config.py::test_import_produces_no_stdout_pollution PASSED    [ 18%]
tests/test_processors.py::test_unicode_nfc_normalization PASSED          [ 21%]
tests/test_processors.py::test_nfc_normalization_preserves_virama PASSED [ 25%]
tests/test_processors.py::test_clean_non_semantic_zero_width_spaces PASSED [ 28%]
tests/test_processors.py::test_preserve_devanagari_zwj_and_zwnj PASSED   [ 31%]
tests/test_processors.py::test_collapse_consecutive_zero_width_joiners PASSED [ 34%]
tests/test_processors.py::test_sentence_segmentation_with_purna_viram PASSED [ 37%]
tests/test_processors.py::test_sentence_segmentation_with_question_and_exclamation PASSED [ 40%]
tests/test_processors.py::test_sentence_segmentation_with_double_danda PASSED [ 43%]
tests/test_processors.py::test_sentence_segmentation_preserves_trailing_sentence_without_delimiter PASSED [ 46%]
tests/test_processors.py::test_sentence_segmentation_preserves_trailing_sentence_after_delimited_one PASSED [ 50%]
tests/test_processors.py::test_sentence_segmentation_with_repeated_delimiters PASSED [ 53%]
tests/test_processors.py::test_sentence_segmentation_with_newlines PASSED [ 56%]
tests/test_processors.py::test_inclusive_min_length_boundary PASSED      [ 59%]
tests/test_processors.py::test_below_min_length_is_excluded PASSED       [ 62%]
tests/test_processors.py::test_inclusive_max_length_boundary PASSED      [ 65%]
tests/test_processors.py::test_devanagari_ratio_pure_nepali PASSED       [ 68%]
tests/test_processors.py::test_devanagari_ratio_code_mixed PASSED        [ 71%]
tests/test_processors.py::test_devanagari_ratio_pure_english PASSED      [ 75%]
tests/test_processors.py::test_empty_string_devanagari_ratio PASSED      [ 78%]
tests/test_processors.py::test_hash_invariance_to_whitespace PASSED      [ 81%]
tests/test_processors.py::test_hash_invariance_to_unicode_decomposition PASSED [ 84%]
tests/test_processors.py::test_distinct_sentences_yield_distinct_hashes PASSED [ 87%]
tests/test_processors.py::test_processor_instance_hash_matches_standalone PASSED [ 90%]
tests/test_processors.py::test_batch_process_segmentation PASSED         [ 93%]
tests/test_processors.py::test_batch_process_normalization PASSED        [ 96%]
tests/test_processors.py::test_processor_statistics PASSED               [100%]

============================= 32 passed in 0.35s ==============================
```

---

## 4. Strict Constraint Adherence Verification

| Constraint | Requirement | Status | Verification Evidence |
|---|---|---|---|
| **Constraint 1** | Do NOT delete or modify legacy synthetic data in `data/` or root | **COMPLIED** | Git status confirms zero modifications to `data/*.jsonl` or `data/*.json`. |
| **Constraint 2** | Do NOT run scrapers against live websites yet | **COMPLIED** | Zero live HTTP collection requests executed. Scraper modules were not called. |
| **Constraint 3** | Do NOT install packages on import | **COMPLIED** | All runtime `os.system("pip install ...")` calls avoided; package imports are clean. |
| **Constraint 4** | Use pytest for all verification | **COMPLIED** | Comprehensive test suite created under `tests/`; all 32 tests verified. |
| **Constraint 5** | All file paths must use `pathlib.Path` relative to repo root | **COMPLIED** | `PROJECT_ROOT = Path(__file__).resolve().parents[1]` used throughout. Zero `d:/` strings. |

---

## 5. Next Step Recommendation (Phase 1.5)

Phase 1 foundations are fully complete, verified, and protected by automated regression tests.

The immediate next logical phase is **Phase 1.5 (Pilot Data Collection & Dataset Validation)**:
1. Implement clean collection adapters for public, accessible sources:
   - News: Public RSS feeds from Ratopati/Setopati.
   - Reviews & Social: Ingest an initial calibration batch of 50–100 genuine records with verified provenance (`REAL_COLLECTED`).
2. Run the newly verified `NepaliTextProcessor` pipeline to segment, clean, validate Devanagari ratio, and assign `sentence_hash`.
3. Produce `DATA_QUALITY_REPORT.md` analyzing real-world vocabulary, sentence length distribution, and domain characteristics before human annotation.

**Stopping here as instructed. Awaiting your approval before proceeding to Phase 1.5.**

