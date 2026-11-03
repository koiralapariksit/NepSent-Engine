"""
Human Annotation Agreement & Adjudication Engine for NepSent-Engine.

Features:
- IAA computation: Raw agreement, Unweighted Cohen's Kappa, Linear Weighted Kappa,
  Quadratic Weighted Kappa, and per-domain breakdown
- Double-blind annotation export (stripping existing labels)
- Human adjudication and consensus resolution pipeline
"""

import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any

from nepali_scraper.config import PROJECT_ROOT, LABEL_SCHEMA, SCHEMA_VERSION

logger = logging.getLogger(__name__)


def compute_confusion_matrix(
    labels_a: List[int],
    labels_b: List[int],
    num_classes: int = 3,
) -> List[List[int]]:
    """
    Compute num_classes x num_classes confusion matrix between two annotators.
    
    Row index i: Annotator A label
    Column index j: Annotator B label
    """
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for a, b in zip(labels_a, labels_b):
        if 0 <= a < num_classes and 0 <= b < num_classes:
            matrix[a][b] += 1
    return matrix


def compute_iaa(
    labels_a: List[int],
    labels_b: List[int],
    domains: Optional[List[str]] = None,
    num_classes: int = 3,
) -> Dict[str, Any]:
    """
    Compute comprehensive Inter-Annotator Agreement (IAA) metrics for ordinal sentiment labels.
    
    Computes:
    - Observed Percentage Agreement (P_o)
    - Unweighted Cohen's Kappa
    - Linear Weighted Cohen's Kappa (penalizing 0 vs 2 twice as heavily as 0 vs 1)
    - Quadratic Weighted Cohen's Kappa
    - Confusion Matrix (3x3)
    - Per-domain agreement breakdown (if domains provided)
    
    Args:
        labels_a: List of integer labels from Annotator A (0, 1, 2)
        labels_b: List of integer labels from Annotator B (0, 1, 2)
        domains: Optional list of domain strings corresponding to each pair
        num_classes: Number of ordinal categories (default: 3)
        
    Returns:
        Dictionary with agreement statistics
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(f"Label length mismatch: {len(labels_a)} vs {len(labels_b)}")
    
    n_samples = len(labels_a)
    if n_samples == 0:
        return {
            "n_samples": 0,
            "observed_agreement": 0.0,
            "cohen_kappa_unweighted": 0.0,
            "cohen_kappa_linear_weighted": 0.0,
            "cohen_kappa_quadratic_weighted": 0.0,
            "confusion_matrix": [[0] * num_classes for _ in range(num_classes)],
            "per_domain": {},
        }
    
    matrix = compute_confusion_matrix(labels_a, labels_b, num_classes=num_classes)
    
    # Marginal distributions
    row_sums = [sum(matrix[i][j] for j in range(num_classes)) for i in range(num_classes)]
    col_sums = [sum(matrix[i][j] for i in range(num_classes)) for j in range(num_classes)]
    
    # 1. Observed Agreement
    agreed_count = sum(matrix[i][i] for i in range(num_classes))
    p_observed = agreed_count / n_samples
    
    # 2. Unweighted Expected Agreement & Cohen's Kappa
    p_expected_unweighted = sum(row_sums[i] * col_sums[i] for i in range(num_classes)) / (n_samples * n_samples)
    if p_expected_unweighted < 1.0:
        kappa_unweighted = (p_observed - p_expected_unweighted) / (1.0 - p_expected_unweighted)
    else:
        kappa_unweighted = 1.0 if p_observed == 1.0 else 0.0
    
    # 3. Linear Weighted Cohen's Kappa
    # Weight w_ij = 1 - |i - j| / (num_classes - 1)
    # For num_classes=3: distance 0 -> 1.0, distance 1 -> 0.5, distance 2 -> 0.0
    max_dist = float(num_classes - 1)
    
    w_observed_linear = sum(
        (1.0 - abs(i - j) / max_dist) * matrix[i][j]
        for i in range(num_classes) for j in range(num_classes)
    ) / n_samples
    
    w_expected_linear = sum(
        (1.0 - abs(i - j) / max_dist) * (row_sums[i] * col_sums[j])
        for i in range(num_classes) for j in range(num_classes)
    ) / (n_samples * n_samples)
    
    if w_expected_linear < 1.0:
        kappa_linear = (w_observed_linear - w_expected_linear) / (1.0 - w_expected_linear)
    else:
        kappa_linear = 1.0 if w_observed_linear == 1.0 else 0.0
    
    # 4. Quadratic Weighted Cohen's Kappa
    # Weight w_ij = 1 - (i - j)^2 / (num_classes - 1)^2
    max_dist_sq = max_dist * max_dist
    
    w_observed_quad = sum(
        (1.0 - float((i - j) ** 2) / max_dist_sq) * matrix[i][j]
        for i in range(num_classes) for j in range(num_classes)
    ) / n_samples
    
    w_expected_quad = sum(
        (1.0 - float((i - j) ** 2) / max_dist_sq) * (row_sums[i] * col_sums[j])
        for i in range(num_classes) for j in range(num_classes)
    ) / (n_samples * n_samples)
    
    if w_expected_quad < 1.0:
        kappa_quad = (w_observed_quad - w_expected_quad) / (1.0 - w_expected_quad)
    else:
        kappa_quad = 1.0 if w_observed_quad == 1.0 else 0.0
    
    # 5. Per-Domain Breakdown
    per_domain: Dict[str, Any] = {}
    if domains and len(domains) == n_samples:
        domain_groups = defaultdict(lambda: {"a": [], "b": []})
        for a, b, d in zip(labels_a, labels_b, domains):
            domain_groups[d]["a"].append(a)
            domain_groups[d]["b"].append(b)
        
        for dom, grp in domain_groups.items():
            per_domain[dom] = compute_iaa(grp["a"], grp["b"], domains=None, num_classes=num_classes)
    
    return {
        "n_samples": n_samples,
        "observed_agreement": round(p_observed, 4),
        "cohen_kappa_unweighted": round(kappa_unweighted, 4),
        "cohen_kappa_linear_weighted": round(kappa_linear, 4),
        "cohen_kappa_quadratic_weighted": round(kappa_quad, 4),
        "confusion_matrix": matrix,
        "label_schema": LABEL_SCHEMA,
        "per_domain": per_domain,
    }


def export_for_annotation(
    records: List[Dict[str, Any]],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """
    Generate independent double-blind annotation sheets for Annotator A and Annotator B.
    
    Strips existing sentiment labels and tags to guarantee independent judgment.
    
    Args:
        records: List of validated records (from DatasetIngestor)
        output_dir: Directory to save annotation files
        
    Returns:
        (path_annotator_a, path_annotator_b)
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    path_a = out_dir / "annotator_A_sheet.jsonl"
    path_b = out_dir / "annotator_B_sheet.jsonl"
    
    def create_blind_item(record: Dict[str, Any], annotator_id: str) -> Dict[str, Any]:
        return {
            "id": record.get("id"),
            "sha256_hash": record.get("sha256_hash"),
            "text": record.get("text"),
            "domain": record.get("domain"),
            "source": record.get("source", "unspecified"),
            "annotator_id": annotator_id,
            "label": None,  # Must be 0 (Negative), 1 (Neutral), or 2 (Positive)
            "confidence": None,  # LOW, MEDIUM, HIGH
            "is_sarcastic": False,
            "notes": "",
        }
    
    with open(path_a, "w", encoding="utf-8") as fa, open(path_b, "w", encoding="utf-8") as fb:
        for r in records:
            item_a = create_blind_item(r, "annotator_A")
            item_b = create_blind_item(r, "annotator_B")
            fa.write(json.dumps(item_a, ensure_ascii=False) + "\n")
            fb.write(json.dumps(item_b, ensure_ascii=False) + "\n")
    
    logger.info(f"Exported {len(records)} blind annotation tasks to {path_a} and {path_b}")
    return path_a, path_b


def adjudicate_annotations(
    file_a: Path,
    file_b: Path,
    adjudicator_resolutions: Optional[Dict[str, int]] = None,
    output_gold_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Merge independent human annotations, compute agreement, and apply adjudicator overrides.
    
    Workflow:
    - If label_A == label_B: Accepted as CONSENSUS.
    - If label_A != label_B and in adjudicator_resolutions: Accepted as ADJUDICATED.
    - If label_A != label_B and not resolved: Flagged as DISAGREED_PENDING.
    
    Args:
        file_a: Path to Annotator A completed JSONL
        file_b: Path to Annotator B completed JSONL
        adjudicator_resolutions: Optional dict mapping record ID to resolved label (0, 1, or 2)
        output_gold_path: Optional Path to save final adjudicated dataset
        
    Returns:
        Summary report containing agreement metrics, consensus counts, and gold records
    """
    if adjudicator_resolutions is None:
        adjudicator_resolutions = {}
    
    def load_sheet(p: Path) -> Dict[str, Dict[str, Any]]:
        lookup = {}
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    item = json.loads(line_str)
                    rec_id = item.get("id") or item.get("sha256_hash")
                    lookup[rec_id] = item
        return lookup
    
    sheet_a = load_sheet(file_a)
    sheet_b = load_sheet(file_b)
    
    common_ids = [rid for rid in sheet_a if rid in sheet_b]
    
    labels_a = []
    labels_b = []
    domains = []
    
    gold_records = []
    disagreements = []
    
    consensus_count = 0
    adjudicated_count = 0
    pending_count = 0
    
    for rid in common_ids:
        item_a = sheet_a[rid]
        item_b = sheet_b[rid]
        
        la = item_a.get("label")
        lb = item_b.get("label")
        dom = item_a.get("domain", "unspecified")
        
        # Valid integer labels
        if la is not None and lb is not None:
            la_int = int(la)
            lb_int = int(lb)
            labels_a.append(la_int)
            labels_b.append(lb_int)
            domains.append(dom)
            
            gold_record = {
                "id": rid,
                "text": item_a.get("text"),
                "domain": dom,
                "sha256_hash": item_a.get("sha256_hash"),
                "source": item_a.get("source"),
                "schema_version": SCHEMA_VERSION,
                "annotator_A_label": la_int,
                "annotator_B_label": lb_int,
                "annotator_A_confidence": item_a.get("confidence"),
                "annotator_B_confidence": item_b.get("confidence"),
                "annotator_A_sarcasm": item_a.get("is_sarcastic", False),
                "annotator_B_sarcasm": item_b.get("is_sarcastic", False),
            }
            
            if la_int == lb_int:
                gold_record["label"] = la_int
                gold_record["adjudication_status"] = "CONSENSUS"
                gold_record["adjudicator_override"] = None
                consensus_count += 1
                gold_records.append(gold_record)
            else:
                disagreement_info = {
                    "id": rid,
                    "text": item_a.get("text"),
                    "domain": dom,
                    "annotator_A": la_int,
                    "annotator_B": lb_int,
                }
                disagreements.append(disagreement_info)
                
                # Check for adjudicator resolution
                if rid in adjudicator_resolutions:
                    res_label = int(adjudicator_resolutions[rid])
                    gold_record["label"] = res_label
                    gold_record["adjudication_status"] = "ADJUDICATED"
                    gold_record["adjudicator_override"] = res_label
                    adjudicated_count += 1
                    gold_records.append(gold_record)
                else:
                    gold_record["label"] = None
                    gold_record["adjudication_status"] = "DISAGREED_PENDING"
                    gold_record["adjudicator_override"] = None
                    pending_count += 1
                    gold_records.append(gold_record)
    
    # Compute agreement
    iaa_metrics = compute_iaa(labels_a, labels_b, domains=domains)
    
    summary = {
        "total_paired_records": len(common_ids),
        "consensus_count": consensus_count,
        "adjudicated_count": adjudicated_count,
        "pending_disagreements_count": pending_count,
        "iaa_metrics": iaa_metrics,
        "disagreements": disagreements,
        "gold_records": gold_records,
    }
    
    # Save gold records if path specified
    if output_gold_path:
        out_p = Path(output_gold_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            for rec in gold_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(gold_records)} adjudicated gold records to {out_p}")
    
    return summary

