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
) -> Dict[str, Any]:
    """
    Partition records into train, validation, and test splits with zero sentence hash leakage.
    
    Guarantees:
    1. Group-Indivisibility: All records with the same sha256_hash are strictly placed
       into the same partition.
    2. Zero Leakage: (train_hashes & val_hashes) == set() and (train_hashes & test_hashes) == set().
    3. Stratification: Partitions preserve the joint distribution of (label, domain).
    4. Statistical Power Check: Emits validator warning if any class in the test set has
       fewer than min_test_support (default: 80) samples.
    
    Args:
        records: List of record dictionaries. Each must contain 'text' or 'sha256_hash',
                 'label', and 'domain'.
        train_ratio: Proportion for training (default 0.70)
        val_ratio: Proportion for validation (default 0.15)
        test_ratio: Proportion for test (default 0.15)
        seed: Random seed for deterministic reproducibility
        min_test_support: Minimum sample count per class in test set
        
    Returns:
        Dictionary containing 'train', 'val', 'test' lists and 'metadata' summary.
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
            "metadata": {
                "total_records": 0,
                "unique_hashes": 0,
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
    
    # 2. Assign each unique hash group to a stratum: (label, domain)
    # If duplicate records have conflicting labels within the same hash, select majority
    stratum_to_hashes: Dict[Tuple[Any, str], List[str]] = defaultdict(list)
    
    for h, group in hash_to_records.items():
        labels = [rec.get("label") for rec in group if rec.get("label") is not None]
        domain = group[0].get("domain", "unspecified")
        
        if labels:
            # Majority label
            majority_label = max(set(labels), key=labels.count)
        else:
            majority_label = "unlabeled"
        
        stratum = (majority_label, domain)
        stratum_to_hashes[stratum].append(h)
    
    rng = random.Random(seed)
    
    train_hashes: Set[str] = set()
    val_hashes: Set[str] = set()
    test_hashes: Set[str] = set()
    
    # 3. Stratified allocation of hash groups
    for stratum, h_list in sorted(stratum_to_hashes.items(), key=lambda x: str(x[0])):
        # Deterministic shuffle within stratum
        shuffled = list(h_list)
        rng.shuffle(shuffled)
        n = len(shuffled)
        
        if n == 1:
            # Single item stratum goes to train
            train_hashes.add(shuffled[0])
        elif n == 2:
            # Two items: one train, one test (or val)
            train_hashes.add(shuffled[0])
            test_hashes.add(shuffled[1])
        else:
            n_train = max(1, int(round(n * train_ratio)))
            n_val = int(round(n * val_ratio))
            # Ensure test gets at least 1 if n >= 3
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
    
    # 4. Strictly verify zero leakage
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
    
    # 5. Materialize records for each partition
    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []
    
    for h in train_hashes:
        train_records.extend(hash_to_records[h])
    for h in val_hashes:
        val_records.extend(hash_to_records[h])
    for h in test_hashes:
        test_records.extend(hash_to_records[h])
    
    # Re-shuffle records within partition deterministically
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)
    
    # 6. Statistical power & sample support checks
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
    
    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(split_result["metadata"], f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved leakage-free splits and metadata to {out_dir}")
    return paths

