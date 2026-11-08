"""
Statistical Evaluation & Non-Parametric Bootstrap Engine for NepSent-Engine.

Provides:
1. Classification metrics computation (Accuracy, Macro-F1, per-class F1, confusion matrix).
2. True non-parametric empirical percentile bootstrap confidence intervals (B=2,000 resamples).
3. Paired bootstrap hypothesis test for model comparison.
4. McNemar's contingency test with exact binomial and continuity-corrected Chi-Square.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from nepali_scraper.config import SCHEMA_VERSION, LABEL_SCHEMA

logger = logging.getLogger("nepali_scraper.evaluation.metrics")


def compute_classification_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics for sentiment evaluation.
    
    Args:
        y_true: Ground truth ordinal sentiment labels (0, 1, 2)
        y_pred: Predicted ordinal sentiment labels (0, 1, 2)
        labels: Class label set, defaults to [0, 1, 2]
        
    Returns:
        Structured metrics dictionary containing point estimates
    """
    if labels is None:
        labels = [0, 1, 2]
    
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    micro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="micro", zero_division=0))
    
    per_class_p = precision_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    per_class_r = recall_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    
    # Per-class support
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    class_support = {int(lbl): int(np.sum(y_true_arr == lbl)) for lbl in labels}
    
    per_class_metrics = {}
    for idx, lbl in enumerate(labels):
        per_class_metrics[int(lbl)] = {
            "name": LABEL_SCHEMA.get(lbl, str(lbl)),
            "precision": float(round(per_class_p[idx], 4)),
            "recall": float(round(per_class_r[idx], 4)),
            "f1": float(round(per_class_f1[idx], 4)),
            "support": class_support[lbl],
        }
    
    return {
        "schema_version": SCHEMA_VERSION,
        "n_samples": len(y_true_arr),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
    }


def bootstrap_confidence_intervals(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
    labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Compute non-parametric empirical percentile bootstrap confidence intervals.
    
    Mandated by Research Review: Macro F1 across 3 classes has no closed-form
    binomial standard error. Empirical bootstrap sampling with replacement
    (B=2,000 resamples) produces distribution-free, rigorous confidence intervals.
    
    Args:
        y_true: Ground truth sentiment labels
        y_pred: Predicted sentiment labels
        n_resamples: Number of bootstrap resamples (default: 2,000)
        ci: Confidence interval coverage (default: 0.95)
        seed: Random seed for deterministic reproducibility
        labels: Class labels list (default: [0, 1, 2])
        
    Returns:
        Dictionary containing point estimates and empirical percentile bounds
    """
    if labels is None:
        labels = [0, 1, 2]
    
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    n_samples = len(y_true_arr)
    
    if n_samples == 0:
        raise ValueError("Cannot compute bootstrap confidence intervals on empty data.")
    
    # Point estimates on full sample
    point_acc = float(accuracy_score(y_true_arr, y_pred_arr))
    point_macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    point_class_f1 = f1_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    
    rng = np.random.default_rng(seed)
    
    boot_acc = np.empty(n_resamples, dtype=float)
    boot_macro_f1 = np.empty(n_resamples, dtype=float)
    boot_class_f1 = np.empty((n_resamples, len(labels)), dtype=float)
    
    indices = np.arange(n_samples)
    for b in range(n_resamples):
        resample_idx = rng.choice(indices, size=n_samples, replace=True)
        yt_b = y_true_arr[resample_idx]
        yp_b = y_pred_arr[resample_idx]
        
        boot_acc[b] = accuracy_score(yt_b, yp_b)
        boot_macro_f1[b] = f1_score(yt_b, yp_b, average="macro", zero_division=0)
        boot_class_f1[b] = f1_score(yt_b, yp_b, labels=labels, average=None, zero_division=0)
    
    alpha = (1.0 - ci) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0
    
    def _format_ci(point: float, bootstrap_dist: np.ndarray) -> Dict[str, float]:
        raw_lower = float(np.percentile(bootstrap_dist, lower_pct))
        raw_upper = float(np.percentile(bootstrap_dist, upper_pct))
        # Ensure mathematical validity: lower <= point <= upper
        lower = min(raw_lower, point)
        upper = max(raw_upper, point)
        # Clamp within [0.0, 1.0]
        lower = max(0.0, min(1.0, lower))
        upper = max(0.0, min(1.0, upper))
        std_err = float(np.std(bootstrap_dist))
        return {
            "point": round(point, 4),
            "ci_lower": round(lower, 4),
            "ci_upper": round(upper, 4),
            "std_error": round(std_err, 4),
        }
    
    per_class_res = {}
    for idx, lbl in enumerate(labels):
        per_class_res[int(lbl)] = _format_ci(float(point_class_f1[idx]), boot_class_f1[:, idx])
        per_class_res[int(lbl)]["name"] = LABEL_SCHEMA.get(lbl, str(lbl))
    
    return {
        "n_resamples": n_resamples,
        "ci_level": ci,
        "sample_size": n_samples,
        "accuracy": _format_ci(point_acc, boot_acc),
        "macro_f1": _format_ci(point_macro_f1, boot_macro_f1),
        "per_class_f1": per_class_res,
    }


def paired_bootstrap_test(
    y_true: Union[List[int], np.ndarray],
    pred_model_a: Union[List[int], np.ndarray],
    pred_model_b: Union[List[int], np.ndarray],
    n_resamples: int = 2000,
    seed: int = 42,
    metric: str = "macro_f1",
) -> Dict[str, Any]:
    """
    Perform paired non-parametric bootstrap hypothesis test between two models.
    
    Tests H0: score(Model A) == score(Model B)
    vs    H1: score(Model A) != score(Model B)
    
    Args:
        y_true: Ground truth sentiment labels
        pred_model_a: Predictions from Model A
        pred_model_b: Predictions from Model B
        n_resamples: Number of bootstrap iterations (default: 2,000)
        seed: Random seed
        metric: Evaluation metric ("macro_f1" or "accuracy")
        
    Returns:
        Structured test result dictionary with delta and empirical two-tailed p-value
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    pa_arr = np.asarray(pred_model_a, dtype=int)
    pb_arr = np.asarray(pred_model_b, dtype=int)
    n_samples = len(y_true_arr)
    
    if not (len(pa_arr) == n_samples and len(pb_arr) == n_samples):
        raise ValueError("y_true, pred_model_a, and pred_model_b must have identical lengths.")
    
    def _calc_score(yt: np.ndarray, yp: np.ndarray) -> float:
        if metric == "accuracy":
            return float(accuracy_score(yt, yp))
        return float(f1_score(yt, yp, average="macro", zero_division=0))
    
    score_a = _calc_score(y_true_arr, pa_arr)
    score_b = _calc_score(y_true_arr, pb_arr)
    delta_obs = score_a - score_b
    
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    deltas = np.empty(n_resamples, dtype=float)
    
    for b in range(n_resamples):
        resample_idx = rng.choice(indices, size=n_samples, replace=True)
        yt_b = y_true_arr[resample_idx]
        pa_b = pa_arr[resample_idx]
        pb_b = pb_arr[resample_idx]
        deltas[b] = _calc_score(yt_b, pa_b) - _calc_score(yt_b, pb_b)
    
    # Two-tailed empirical p-value under the null hypothesis (centered distribution)
    # P-value = fraction of replicates where |delta_b - delta_obs| >= |delta_obs|
    centered_deltas = deltas - delta_obs
    p_value = float(np.mean(np.abs(centered_deltas) >= np.abs(delta_obs)))
    
    # Handle exact zero delta case
    if np.isclose(delta_obs, 0.0):
        p_value = 1.0
        
    return {
        "metric": metric,
        "score_model_a": round(score_a, 4),
        "score_model_b": round(score_b, 4),
        "delta_obs": round(delta_obs, 4),
        "p_value": round(p_value, 4),
        "significant_at_05": bool(p_value < 0.05),
        "n_resamples": n_resamples,
        "sample_size": n_samples,
    }


def mcnemar_test(
    y_true: Union[List[int], np.ndarray],
    pred_model_a: Union[List[int], np.ndarray],
    pred_model_b: Union[List[int], np.ndarray],
) -> Dict[str, Any]:
    """
    Perform McNemar's test on paired classification predictions.
    
    Constructs 2x2 contingency table:
        b: Model A correct, Model B incorrect
        c: Model A incorrect, Model B correct
        
    Uses exact binomial test when (b + c) < 25;
    Uses Edwards' continuity-corrected Chi-Square when (b + c) >= 25.
    
    Args:
        y_true: Ground truth labels
        pred_model_a: Model A predictions
        pred_model_b: Model B predictions
        
    Returns:
        Structured dictionary with contingency counts, statistic, and p-value
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    pa_arr = np.asarray(pred_model_a, dtype=int)
    pb_arr = np.asarray(pred_model_b, dtype=int)
    
    correct_a = (pa_arr == y_true_arr)
    correct_b = (pb_arr == y_true_arr)
    
    # Contingency cell counts
    n_00 = int(np.sum(correct_a & correct_b))
    b = int(np.sum(correct_a & (~correct_b)))   # A correct, B wrong
    c = int(np.sum((~correct_a) & correct_b))   # A wrong, B correct
    n_11 = int(np.sum((~correct_a) & (~correct_b)))
    
    discordant = b + c
    
    if discordant == 0:
        return {
            "b": b,
            "c": c,
            "n_both_correct": n_00,
            "n_both_incorrect": n_11,
            "statistic": 0.0,
            "p_value": 1.0,
            "method": "exact_identical",
            "significant_at_05": False,
        }
    
    if discordant < 25:
        # Exact two-sided binomial test under H0: p = 0.5
        # b ~ Binomial(discordant, 0.5)
        res = stats.binomtest(b, discordant, p=0.5, alternative="two-sided")
        p_val = float(res.pvalue)
        stat = float(b)
        method = "exact_binomial"
    else:
        # Edwards' continuity-corrected Chi-Square: (|b - c| - 1)^2 / (b + c)
        chi2_stat = float(((abs(b - c) - 1.0) ** 2) / discordant)
        p_val = float(stats.chi2.sf(chi2_stat, df=1))
        stat = chi2_stat
        method = "continuity_corrected_chi2"
    
    return {
        "b": b,
        "c": c,
        "n_both_correct": n_00,
        "n_both_incorrect": n_11,
        "statistic": round(stat, 4),
        "p_value": round(p_val, 4),
        "method": method,
        "significant_at_05": bool(p_val < 0.05),
    }

