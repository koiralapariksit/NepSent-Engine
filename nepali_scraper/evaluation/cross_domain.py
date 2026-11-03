"""
Cross-Domain and Leave-One-Domain-Out (LODO) Generalization Evaluator for NepSent-Engine.

Provides:
1. Per-domain performance slice evaluation with statistical power safeguards (N < 50 warning).
2. True Leave-One-Domain-Out cross-validation across distinct genre domains (e.g., news, reviews, social).
"""

import logging
from typing import List, Dict, Any, Callable, Optional, Union
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from nepali_scraper.config import SCHEMA_VERSION, LABEL_SCHEMA
from nepali_scraper.evaluation.metrics import compute_classification_metrics

logger = logging.getLogger("nepali_scraper.evaluation.cross_domain")


def evaluate_per_domain(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    domains: List[str],
    labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Evaluate sentiment metrics disaggregated by text domain.
    
    Automatically flags domain slices with small sample sizes (N < 50) as exploratory,
    as required by the statistical power guidelines.
    
    Args:
        y_true: Ground truth sentiment labels
        y_pred: Predicted sentiment labels
        domains: Domain identifier list corresponding to each sample
        labels: Class label set, defaults to [0, 1, 2]
        
    Returns:
        Structured dictionary with per-domain metrics and sample counts
    """
    if labels is None:
        labels = [0, 1, 2]
    
    yt_arr = np.asarray(y_true, dtype=int)
    yp_arr = np.asarray(y_pred, dtype=int)
    n_total = len(yt_arr)
    
    if not (len(yp_arr) == n_total and len(domains) == n_total):
        raise ValueError(
            f"Dimension mismatch: y_true ({n_total}), y_pred ({len(yp_arr)}), "
            f"and domains ({len(domains)}) must have identical length."
        )
    
    unique_domains = sorted(list(set(domains)))
    domain_results = {}
    
    for dom in unique_domains:
        mask = [d == dom for d in domains]
        dom_yt = yt_arr[mask]
        dom_yp = yp_arr[mask]
        n_dom = len(dom_yt)
        
        is_exploratory = bool(n_dom < 50)
        warning_msg = (
            f"Statistical power warning: domain '{dom}' has N={n_dom} < 50; "
            f"variance is high and results must be treated as exploratory."
            if is_exploratory else None
        )
        
        dom_metrics = compute_classification_metrics(dom_yt, dom_yp, labels=labels)
        
        domain_results[dom] = {
            "n_samples": n_dom,
            "accuracy": dom_metrics["accuracy"],
            "macro_f1": dom_metrics["macro_f1"],
            "weighted_f1": dom_metrics["weighted_f1"],
            "is_exploratory": is_exploratory,
            "warning": warning_msg,
            "per_class": dom_metrics["per_class"],
            "confusion_matrix": dom_metrics["confusion_matrix"],
        }
    
    overall = compute_classification_metrics(yt_arr, yp_arr, labels=labels)
    
    return {
        "schema_version": SCHEMA_VERSION,
        "total_samples": n_total,
        "overall_accuracy": overall["accuracy"],
        "overall_macro_f1": overall["macro_f1"],
        "domain_breakdown": domain_results,
    }


def run_lodo_evaluation(
    records: List[Dict[str, Any]],
    model_factory: Callable[[], Any],
    text_key: str = "text",
    label_key: str = "label",
    domain_key: str = "domain",
    labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Execute Leave-One-Domain-Out (LODO) cross-domain generalization evaluation.
    
    Iterates through all unique domains: in each fold, the model is trained
    on (D - 1) domains and evaluated exclusively on the held-out domain.
    
    Args:
        records: List of sample dictionaries containing text, label, and domain
        model_factory: Callable returning a fresh, un-fitted model instance
                       with .fit(texts, labels) and .predict(texts) methods
        text_key: Dictionary key for text string
        label_key: Dictionary key for ordinal label (0, 1, 2)
        domain_key: Dictionary key for domain identifier string
        labels: Class label set, defaults to [0, 1, 2]
        
    Returns:
        Structured dictionary containing per-domain held-out results and LODO macro averages
    """
    if labels is None:
        labels = [0, 1, 2]
    
    all_domains = sorted(list(set(r[domain_key] for r in records)))
    
    if len(all_domains) < 2:
        raise ValueError(
            f"Leave-One-Domain-Out evaluation requires at least 2 distinct domains; "
            f"found {len(all_domains)}: {all_domains}"
        )
    
    fold_results = {}
    macro_f1_list = []
    accuracy_list = []
    
    for held_out_domain in all_domains:
        train_records = [r for r in records if r[domain_key] != held_out_domain]
        test_records = [r for r in records if r[domain_key] == held_out_domain]
        
        train_texts = [r[text_key] for r in train_records]
        train_labels = [int(r[label_key]) for r in train_records]
        test_texts = [r[text_key] for r in test_records]
        test_labels = [int(r[label_key]) for r in test_records]
        
        n_test = len(test_labels)
        is_exploratory = bool(n_test < 50)
        
        # Instantiate fresh model and fit on (D - 1) domains
        model = model_factory()
        model.fit(train_texts, train_labels)
        
        # Predict on held-out domain
        predictions = model.predict(test_texts)
        
        metrics = compute_classification_metrics(test_labels, predictions, labels=labels)
        
        warning_msg = (
            f"Statistical power warning: Held-out domain '{held_out_domain}' has N={n_test} < 50."
            if is_exploratory else None
        )
        
        fold_results[held_out_domain] = {
            "held_out_domain": held_out_domain,
            "train_samples": len(train_labels),
            "test_samples": n_test,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "is_exploratory": is_exploratory,
            "warning": warning_msg,
            "per_class": metrics["per_class"],
            "confusion_matrix": metrics["confusion_matrix"],
        }
        
        macro_f1_list.append(metrics["macro_f1"])
        accuracy_list.append(metrics["accuracy"])
    
    mean_macro_f1 = float(np.mean(macro_f1_list))
    mean_accuracy = float(np.mean(accuracy_list))
    
    return {
        "schema_version": SCHEMA_VERSION,
        "n_domains": len(all_domains),
        "mean_lodo_macro_f1": round(mean_macro_f1, 4),
        "mean_lodo_accuracy": round(mean_accuracy, 4),
        "per_domain_results": fold_results,
    }

