# Legacy V1 Architecture Archive (Historical Audit Artifacts)

> [!WARNING]
> **DO NOT USE SCRIPTS IN THIS DIRECTORY FOR PRODUCTION OR BENCHMARKING.**
> The files in this archive represent the legacy v1 codebase of NepSent-Engine prior to the Phase Zero through Phase 4.5 scientific research upgrade.

## Why Were These Files Archived?

During the Phase Zero independent scientific and software architecture audit, several critical methodological flaws were uncovered in the v1 scripts:

1. **Synthetic Data Inflation & Simulators:**
   - `phase2_annotation_simulator.py` and `generate_test_data.py` used heuristic rules and probabilistic random generators to manufacture fake "annotator" labels and synthetic data, which invalidated inter-annotator agreement and inflated benchmark metrics.
2. **Schema Inversion Flaw:**
   - The v1 schema used nominal mapping (`0: Negative, 1: Positive, 2: Neutral`), which broke distance assumptions when computing weighted Cohen's Kappa ($\kappa_w$), penalizing a Negative vs. Neutral mismatch more severely than Negative vs. Positive. (This was corrected in modern v2.0 to `0: Negative, 1: Neutral, 2: Positive`).
3. **Data Leakage in Splitting:**
   - `phase2_data_split.py` applied random train/test splits without sentence hash grouping, causing exact and near-duplicate sentences across web sources to leak into the test set.
4. **Hardcoded Drive Paths & Network Side-Effects on Import:**
   - Multiple v1 scripts contained hardcoded absolute Windows paths (`d:/...`) and initiated network requests or unbuffered print statements during import.

## Modern Replacement

All research workflows have been replaced by the modular, tested, leakage-free `nepali_scraper` package and unified CLI:

| Legacy Script (Archived) | Modern Replacement (Active) |
| :--- | :--- |
| `generate_test_data.py` / `process_pipeline.py` | `nepali_scraper/collectors.py` (`nepsent ingest`) |
| `phase2_data_split.py` | `nepali_scraper/splitter.py` (`nepsent split`) |
| `phase2_annotation_simulator.py` / `phase2_kappa_calculation.py` | `nepali_scraper/annotation.py` (`nepsent iaa`) |
| `phase3_model_trainer.py` / `phase3_tfidf_baseline.py` | `nepali_scraper/models/` (`nepsent train`) |
| `phase3_error_analysis.py` | `nepali_scraper/evaluation/` (`nepsent evaluate`) |
| `classify_nepali.py` | `nepali_scraper/cli.py` (`nepsent predict`) |

These legacy files are retained strictly for archival transparency, git history integrity, and comparative auditability.

