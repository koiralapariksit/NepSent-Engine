"""
Evaluation & Linguistic Error Taxonomy Subsystem for NepSent-Engine.

Exports:
- Statistical evaluation metrics and non-parametric bootstrap confidence intervals.
- Paired bootstrap and McNemar's hypothesis testing for model comparisons.
- Per-domain evaluation and Leave-One-Domain-Out (LODO) cross-validation.
- Nepali morphological negation detector, code-mixing analyzer, subword fertility calculator,
  and diagnostic error taxonomy generator.
"""

from nepali_scraper.evaluation.metrics import (
    compute_classification_metrics,
    bootstrap_confidence_intervals,
    paired_bootstrap_test,
    mcnemar_test,
)
from nepali_scraper.evaluation.cross_domain import (
    evaluate_per_domain,
    run_lodo_evaluation,
)
from nepali_scraper.evaluation.linguistic_analysis import (
    detect_nepali_negation,
    analyze_code_mixing,
    compute_subword_fertility,
    generate_error_taxonomy,
)

__all__ = [
    "compute_classification_metrics",
    "bootstrap_confidence_intervals",
    "paired_bootstrap_test",
    "mcnemar_test",
    "evaluate_per_domain",
    "run_lodo_evaluation",
    "detect_nepali_negation",
    "analyze_code_mixing",
    "compute_subword_fertility",
    "generate_error_taxonomy",
]

