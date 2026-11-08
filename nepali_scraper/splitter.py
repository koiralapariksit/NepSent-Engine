"""
Leakage-free, group-stratified dataset partitioner for NepSent-Engine.

Features:
- Guaranteed zero sentence-hash overlap across train, validation, and test splits
- Stratified group partitioning by (label, domain)
- Statistical power validation (warnings when test class support < 80 samples)
"""

import json
import random
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional, Any

from nepali_scraper.processors import compute_sentence_hash

logger = logging.getLogger(__name__)

MIN_RECOMMENDED_TEST_SUPPORT = 80


def hash_grouped_stratified_split(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    min_test_support: int = MIN_RECOMMENDED_TEST_SUPPORT,
    collapse_unanimous: bool = True,
) -> Dict[str, Any]:
    """
    Partition records into train, validation, and test splits with zero sentence hash leakage.
    
    Guarantees:
    1. Group-Indivisibility: All records with the same sha256_hash are strictly placed
       into the same partition.
    2. Zero Leakage: (train_hashes & val_hashes) == set() and (train_hashes & test_hashes) == set().
    3. Contradictory Duplicate Quarantine: If a hash cluster has conflicting labels without
       prior adjudication, it is excluded from splits and returned in 'conflicts'.
    4. Unanimous Duplicate Collapsing: If a hash cluster has unanimous labels and collapse_unanimous=True,
       it is collapsed into 1 canonical record with merged 'provenance_sources'.
    5. Graceful Stratification Fallback: Falls back to label-only stratification if any
       composite (label, domain) cell has < 2 samples.
    6. Statistical Power Check: Emits validator warning if any class in the test set has
       fewer than min_test_support (default: 80) samples.
    
    Args:
        records: List of record dictionaries. Each must contain 'text' or 'sha256_hash',
                 'label', and 'domain'.
        train_ratio: Proportion for training (default 0.70)
        val_ratio: Proportion for validation (default 0.15)
        test_ratio: Proportion for test (default 0.15)
        seed: Random seed for deterministic reproducibility
        min_test_support: Minimum sample count per class in test set
        collapse_unanimous: Whether to collapse unanimous duplicates into 1 canonical record
        
    Returns:
        Dictionary containing 'train', 'val', 'test', 'conflicts' lists and 'metadata' summary.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
        raise ValueError(
            f"Ratios must sum to 1.0 (got {train_ratio} + {val_ratio} + {test_ratio} = "
            f"{train_ratio + val_ratio + test_ratio})"
        )
    
    if not records:
        return {
            "train": [],
            "val": [],
            "test": [],
            "conflicts": [],
            "metadata": {
                "total_records": 0,
                "unique_hashes": 0,
                "conflicts_count": 0,
                "warnings": ["Empty records input"],
            }
        }
    
    # 1. Group records by unique sentence hash
    hash_to_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        h = r.get("sha256_hash")
        if not h:
            text = r.get("text", "")
            h = compute_sentence_hash(text)
            r["sha256_hash"] = h
        hash_to_records[h].append(r)
    
    # 2. Contradictory Duplicate Quarantine & Unanimous Collapsing
    conflicts: List[Dict[str, Any]] = []
    clean_hash_to_records: Dict[str, List[Dict[str, Any]]] = {}
    
    for h, group in hash_to_records.items():
        labels = [rec.get("label") for rec in group if rec.get("label") is not None]
        unique_labels = set(labels)
        is_adjudicated = any(
            rec.get("is_adjudicated") is True or rec.get("adjudication_status") == "ADJUDICATED"
            for rec in group
        )
        
        # Invariant B1: If cluster has conflicting labels and is not adjudicated, exclude all records
        if len(unique_labels) > 1 and not is_adjudicated:
            for rec in group:
                conflict_item = dict(rec)
                conflict_item["conflict_reason"] = f"Contradictory duplicate labels: {sorted(list(unique_labels))}"
                conflicts.append(conflict_item)
            continue
            
        # Invariant B1: If cluster has unanimous labels, collapse to 1 canonical record and merge provenance_sources
        if len(group) > 1 and collapse_unanimous:
            canonical = dict(group[0])
            all_sources = []
            for rec in group:
                src = rec.get("source")
                if src:
                    if isinstance(src, list):
                        all_sources.extend(src)
                    else:
                        all_sources.append(src)
                if "provenance_sources" in rec and isinstance(rec["provenance_sources"], list):
                    all_sources.extend(rec["provenance_sources"])
            canonical["provenance_sources"] = sorted(list(set(all_sources))) if all_sources else ["UNVERIFIED"]
            clean_hash_to_records[h] = [canonical]
        else:
            clean_hash_to_records[h] = group
            
    # 3. Stratification with fallback
    composite_strata: Dict[Tuple[Any, str], List[str]] = defaultdict(list)
    label_only_strata: Dict[Any, List[str]] = defaultdict(list)
    
    for h, group in clean_hash_to_records.items():
        rec = group[0]
        lbl = rec.get("label")
        label_key = lbl if lbl is not None else "unlabeled"
        domain = rec.get("domain", "unspecified")
        
        composite_strata[(label_key, domain)].append(h)
        label_only_strata[label_key].append(h)
        
    # Check if any composite cell has < 2 samples
    has_sparse_composite = any(len(h_list) < 2 for h_list in composite_strata.values())
    if has_sparse_composite:
        logger.info(
            "Composite (label, domain) stratification has cells with < 2 samples; "
            "falling back to label-only stratification."
        )
        selected_strata = label_only_strata
        stratification_mode = "label_only_fallback"
    else:
        selected_strata = composite_strata
        stratification_mode = "composite_label_domain"
    
    rng = random.Random(seed)
    
    train_hashes: Set[str] = set()
    val_hashes: Set[str] = set()
    test_hashes: Set[str] = set()
    
    # 4. Stratified allocation of hash groups
    for stratum_key, h_list in sorted(selected_strata.items(), key=lambda x: str(x[0])):
        shuffled = list(h_list)
        rng.shuffle(shuffled)
        n = len(shuffled)
        
        if n == 1:
            train_hashes.add(shuffled[0])
        elif n == 2:
            train_hashes.add(shuffled[0])
            test_hashes.add(shuffled[1])
        else:
            n_train = max(1, int(round(n * train_ratio)))
            n_val = int(round(n * val_ratio))
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
            n_test = n - n_train - n_val
            
            for h in shuffled[:n_train]:
                train_hashes.add(h)
            for h in shuffled[n_train:n_train + n_val]:
                val_hashes.add(h)
            for h in shuffled[n_train + n_val:]:
                test_hashes.add(h)
    
    # 5. Strictly verify zero leakage
    overlap_train_val = train_hashes & val_hashes
    overlap_train_test = train_hashes & test_hashes
    overlap_val_test = val_hashes & test_hashes
    
    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise RuntimeError(
            f"CRITICAL LEAKAGE DETECTED in hash_grouped_stratified_split: "
            f"train&val={len(overlap_train_val)}, "
            f"train&test={len(overlap_train_test)}, "
            f"val&test={len(overlap_val_test)}"
        )
    
    # 6. Materialize records for each partition
    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []
    
    for h in train_hashes:
        train_records.extend(clean_hash_to_records[h])
    for h in val_hashes:
        val_records.extend(clean_hash_to_records[h])
    for h in test_hashes:
        test_records.extend(clean_hash_to_records[h])
    
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)
    
    # 7. Statistical power & sample support checks
    warnings: List[str] = []
    
    test_class_counts: Dict[Any, int] = defaultdict(int)
    for r in test_records:
        lbl = r.get("label")
        if lbl is not None:
            test_class_counts[lbl] += 1
    
    for lbl, count in sorted(test_class_counts.items(), key=lambda x: str(x[0])):
        if count < min_test_support:
            warning_msg = (
                f"Statistical Power Warning: Test partition class '{lbl}' has only {count} samples "
                f"(recommended minimum: >= {min_test_support}). Cochran's formula and BCa bootstrap "
                f"confidence intervals require >= {min_test_support} samples per class for stable Macro F1 bounds."
            )
            warnings.append(warning_msg)
            logger.warning(warning_msg)
    
    metadata = {
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "total_records": len(records),
        "total_unique_hashes": len(hash_to_records),
        "quarantined_conflicts_count": len(conflicts),
        "stratification_mode": stratification_mode,
        "split_counts": {
            "train": {"records": len(train_records), "unique_hashes": len(train_hashes)},
            "val": {"records": len(val_records), "unique_hashes": len(val_hashes)},
            "test": {"records": len(test_records), "unique_hashes": len(test_hashes)},
        },
        "test_class_distribution": dict(test_class_counts),
        "min_test_support_threshold": min_test_support,
        "warnings": warnings,
        "zero_leakage_verified": True,
    }
    
    return {
        "train": train_records,
        "val": val_records,
        "test": test_records,
        "conflicts": conflicts,
        "metadata": metadata,
    }


def save_splits(
    split_result: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Save train, validation, test splits and metadata JSON to disk.
    
    Args:
        split_result: Output dictionary from hash_grouped_stratified_split
        output_dir: Target directory path
        
    Returns:
        Dictionary mapping partition names to output file Paths
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {
        "train": out_dir / "05_train_split.jsonl",
        "val": out_dir / "03_validation_partition.jsonl",
        "test": out_dir / "06_test_split.jsonl",
        "metadata": out_dir / "split_metadata.json",
    }
    
    for part in ["train", "val", "test"]:
        with open(paths[part], "w", encoding="utf-8") as f:
            for rec in split_result[part]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    if split_result.get("conflicts"):
        conflicts_file = out_dir / "conflicts.jsonl"
        paths["conflicts"] = conflicts_file
        with open(conflicts_file, "w", encoding="utf-8") as f:
            for rec in split_result["conflicts"]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"Quarantined {len(split_result['conflicts'])} conflicting records to {conflicts_file}")
    
    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(split_result["metadata"], f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved leakage-free splits and metadata to {out_dir}")
    return paths

