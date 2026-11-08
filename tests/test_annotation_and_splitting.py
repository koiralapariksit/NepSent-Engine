"""
Unit tests for NepSent-Engine Phase 1.5 & Phase 2 components:
- DatasetIngestor (exact & near-duplicate removal, Hindi contamination detection)
- Annotation Agreement (unweighted vs linear/quadratic weighted Kappa math, per-domain breakdown)
- Double-blind export & human adjudication engine
- Leakage-free, hash-grouped stratified splitter (zero hash overlap, power warnings)
"""

import sys
import json
import pytest
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nepali_scraper.collectors import (
    DatasetIngestor,
    detect_hindi_contamination,
    char_ngram_jaccard,
)
from nepali_scraper.annotation import (
    compute_iaa,
    export_for_annotation,
    adjudicate_annotations,
)
from nepali_scraper.splitter import (
    hash_grouped_stratified_split,
)


# ============================================================================
# 1. DATA INGESTION, PROVENANCE & DEDUPLICATION TESTS
# ============================================================================

def test_exact_deduplication_via_sha256():
    """Verify that exact duplicate sentences (even with whitespace differences) are rejected."""
    ingestor = DatasetIngestor(min_devanagari_ratio=0.70)
    
    records = [
        {
            "id": "rec_001",
            "text": "नेपालको अर्थतन्त्र सुधारको बाटोमा अग्रसर हुँदैछ।",
            "domain": "news",
            "source": "ratopati_rss",
            "license": "CC-BY-4.0",
        },
        {
            "id": "rec_002",
            "text": "   नेपालको   अर्थतन्त्र   सुधारको   बाटोमा   अग्रसर हुँदैछ।   ",
            "domain": "news_politics",
            "source": "setopati",
            "license": "CC-BY-4.0",
        },
    ]
    
    accepted = ingestor.ingest_records(records)
    assert len(accepted) == 1
    assert ingestor.stats["accepted"] == 1
    assert ingestor.stats["rejected_exact_duplicate"] == 1


def test_near_duplicate_detection_via_jaccard():
    """Verify that sentences with trivial character changes (>= 0.85 Jaccard) are caught as near-duplicates."""
    ingestor = DatasetIngestor(near_duplicate_threshold=0.85)
    
    # Sentence 1 and Sentence 2 differ only by minor punctuation/whitespace
    records = [
        {
            "id": "rec_001",
            "text": "काठमाडौं उपत्यकामा आज बिहानैदेखि मूसलधार पानी परेको छ।",
            "domain": "news",
            "source": "onlinekhabar",
            "license": "CC-BY-4.0",
        },
        {
            "id": "rec_002",
            "text": "काठमाडौं उपत्यकामा आज बिहानैदेखि मूसलधार पानी परेको छ!",  # Only trailing delimiter changed
            "domain": "news",
            "source": "ratopati",
            "license": "CC-BY-4.0",
        },
        {
            "id": "rec_003",
            "text": "पोखरामा पर्यटकहरूको आगमन उल्लेखनीय रूपमा बढेको पाइएको छ।",  # Distinct sentence
            "domain": "news",
            "source": "setopati",
            "license": "CC-BY-4.0",
        },
    ]
    
    accepted = ingestor.ingest_records(records)
    assert len(accepted) == 2
    assert ingestor.stats["accepted"] == 2
    assert ingestor.stats["rejected_near_duplicate"] == 1


def test_hindi_contamination_detection():
    """Verify that sentences with Hindi-exclusive function words are flagged and dropped."""
    # Authentic Nepali sentence
    nepali_clean = "सरकारले नयाँ शिक्षा नीति घोषणा गरेको छ र यो प्रभावकारी हुनेछ।"
    is_contam_clean, markers_clean = detect_hindi_contamination(nepali_clean)
    assert not is_contam_clean
    assert len(markers_clean) == 0
    
    # Contaminated sentence containing Hindi markers: "है", "नहीं", "था"
    hindi_contaminated = "यह सरकार की नई नीति है और इसमें कुछ अच्छा नहीं था।"
    is_contam_dirty, markers_dirty = detect_hindi_contamination(hindi_contaminated)
    assert is_contam_dirty
    assert "है" in markers_dirty
    assert "नहीं" in markers_dirty
    assert "था" in markers_dirty
    
    # Sentence with colloquial Hindi loanword ('लेकिन') but authentic Nepali verbal inflections ('छ', 'भयो')
    nepali_with_loanword = "यो सामान धेरै राम्रो छ लेकिन मूल्य अलि महँगो भयो।"
    is_contam_loan, markers_loan = detect_hindi_contamination(nepali_with_loanword)
    assert not is_contam_loan
    assert "लेकिन" in markers_loan

    # Ingestor drops contaminated sentence by default
    ingestor = DatasetIngestor(drop_hindi_contaminated=True)
    records = [
        {"text": nepali_clean, "domain": "news", "source": "test", "license": "CC-0"},
        {"text": nepali_with_loanword, "domain": "news", "source": "test", "license": "CC-0"},
        {"text": hindi_contaminated, "domain": "news", "source": "test", "license": "CC-0"},
    ]
    accepted = ingestor.ingest_records(records)
    assert len(accepted) == 2
    assert ingestor.stats["flagged_hindi_contamination"] == 1


def test_devanagari_ratio_enforcement():
    """Verify that records with Devanagari ratio < 0.70 are rejected."""
    ingestor = DatasetIngestor(min_devanagari_ratio=0.70)
    
    # Mostly English text
    mostly_english = "The price of this item is too high in नेपाल।"  # ~15% Devanagari
    # Mostly Devanagari with acceptable English word
    mostly_devanagari = "यो नयाँ laptop को गति र ब्याट्री लाइफ एकदम राम्रो छ।"  # >80% Devanagari
    
    records = [
        {"text": mostly_english, "domain": "reviews", "source": "test", "license": "CC-0"},
        {"text": mostly_devanagari, "domain": "reviews", "source": "test", "license": "CC-0"},
    ]
    accepted = ingestor.ingest_records(records)
    assert len(accepted) == 1
    assert accepted[0]["text"] == mostly_devanagari
    assert ingestor.stats["rejected_devanagari_ratio"] == 1


# ============================================================================
# 2. ANNOTATION AGREEMENT (IAA) & ORDINAL WEIGHTING TESTS
# ============================================================================

def test_perfect_iaa_agreement():
    """Verify that identical annotations produce kappa = 1.0 across all metrics."""
    labels = [0, 0, 1, 1, 2, 2, 0, 1, 2, 0]
    res = compute_iaa(labels, labels)
    assert res["observed_agreement"] == 1.0
    assert res["cohen_kappa_unweighted"] == 1.0
    assert res["cohen_kappa_linear_weighted"] == 1.0
    assert res["cohen_kappa_quadratic_weighted"] == 1.0


def test_ordinal_weighted_kappa_penalizes_extreme_mismatches_more():
    """
    CRITICAL METHODOLOGICAL TEST:
    Verify that an extreme disagreement (0 vs 2, error distance 2) is penalized
    more heavily under Linear Weighted Kappa than an adjacent disagreement (0 vs 1, distance 1).
    """
    base = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    
    # Case A: 2 disagreements, but both are adjacent: (0 vs 1) and (1 vs 2)
    # Distance = 1 each
    adjacent_disagreements = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    res_adjacent = compute_iaa(base, adjacent_disagreements)
    
    # Case B: 2 disagreements, but both are extreme: (0 vs 2) and (2 vs 0)
    # Distance = 2 each
    extreme_disagreements = [0, 0, 0, 2, 1, 1, 1, 1, 2, 2, 2, 0]
    res_extreme = compute_iaa(base, extreme_disagreements)
    
    # Both cases have the exact same number of disagreements (2 out of 12)
    assert res_adjacent["observed_agreement"] == res_extreme["observed_agreement"]
    
    # Under Linear Weighted Kappa, adjacent disagreements MUST have a higher kappa than extreme disagreements
    assert res_adjacent["cohen_kappa_linear_weighted"] > res_extreme["cohen_kappa_linear_weighted"]
    # Quadratic weighted kappa must also be higher for adjacent errors
    assert res_adjacent["cohen_kappa_quadratic_weighted"] > res_extreme["cohen_kappa_quadratic_weighted"]


def test_iaa_per_domain_breakdown():
    """Verify that per-domain agreement breakdown correctly computes domain-specific metrics."""
    labels_a = [0, 1, 2, 0, 1, 2]
    labels_b = [0, 1, 2, 2, 1, 0]  # First 3 agree (domain news), last 3 disagree (domain social)
    domains = ["news", "news", "news", "social", "social", "social"]
    
    res = compute_iaa(labels_a, labels_b, domains=domains)
    assert "news" in res["per_domain"]
    assert "social" in res["per_domain"]
    assert res["per_domain"]["news"]["observed_agreement"] == 1.0
    assert res["per_domain"]["social"]["observed_agreement"] < 1.0


# ============================================================================
# 3. DOUBLE-BLIND EXPORT & ADJUDICATION TESTS
# ============================================================================

def test_export_for_annotation_blinds_labels(tmp_path):
    """Verify that export_for_annotation creates blind sheets without leaking original labels."""
    records = [
        {
            "id": "rec_001",
            "sha256_hash": "hash001",
            "text": "नेपाल एक सुन्दर देश हो।",
            "domain": "news_politics",
            "source": "ratopati",
            "label": 2,  # Existing ground truth
        }
    ]
    
    path_a, path_b = export_for_annotation(records, tmp_path)
    assert path_a.is_file()
    assert path_b.is_file()
    
    with open(path_a, "r", encoding="utf-8") as f:
        item = json.loads(f.readline())
        assert item["label"] is None  # Must be stripped
        assert item["annotator_id"] == "annotator_A"
        assert item["text"] == "नेपाल एक सुन्दर देश हो।"
    
    with open(path_b, "r", encoding="utf-8") as f:
        item = json.loads(f.readline())
        assert item["label"] is None  # Must be stripped
        assert item["annotator_id"] == "annotator_B"


def test_adjudicate_annotations_workflow(tmp_path):
    """Verify that consensus items are accepted and disagreed items apply adjudicator override."""
    # Write Annotator A sheet
    file_a = tmp_path / "ann_a.jsonl"
    with open(file_a, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "1", "sha256_hash": "h1", "text": "वाक्य १", "domain": "news", "label": 2}) + "\n")
        f.write(json.dumps({"id": "2", "sha256_hash": "h2", "text": "वाक्य २", "domain": "news", "label": 0}) + "\n")
        f.write(json.dumps({"id": "3", "sha256_hash": "h3", "text": "वाक्य ३", "domain": "news", "label": 1}) + "\n")
    
    # Write Annotator B sheet
    file_b = tmp_path / "ann_b.jsonl"
    with open(file_b, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "1", "sha256_hash": "h1", "text": "वाक्य १", "domain": "news", "label": 2}) + "\n")  # Agreed (2,2)
        f.write(json.dumps({"id": "2", "sha256_hash": "h2", "text": "वाक्य २", "domain": "news", "label": 2}) + "\n")  # Disagreed (0,2) -> Override to 0
        f.write(json.dumps({"id": "3", "sha256_hash": "h3", "text": "वाक्य ३", "domain": "news", "label": 0}) + "\n")  # Disagreed (1,0) -> Pending
    
    gold_path = tmp_path / "gold.jsonl"
    resolutions = {"2": 0}  # Adjudicator overrides item 2 to 0
    
    summary = adjudicate_annotations(file_a, file_b, adjudicator_resolutions=resolutions, output_gold_path=gold_path)
    
    assert summary["consensus_count"] == 1
    assert summary["adjudicated_count"] == 1
    assert summary["pending_disagreements_count"] == 1
    assert gold_path.is_file()
    
    gold_by_id = {rec["id"]: rec for rec in summary["gold_records"]}
    assert gold_by_id["1"]["adjudication_status"] == "CONSENSUS"
    assert gold_by_id["1"]["label"] == 2
    assert gold_by_id["2"]["adjudication_status"] == "ADJUDICATED"
    assert gold_by_id["2"]["label"] == 0
    assert gold_by_id["3"]["adjudication_status"] == "DISAGREED_PENDING"
    assert gold_by_id["3"]["label"] is None


# ============================================================================
# 4. LEAKAGE-FREE STRATIFIED SPLITTER TESTS
# ============================================================================

def test_hash_grouped_stratified_split_zero_leakage():
    """
    CRITICAL METHODOLOGICAL TEST:
    Verify that identical sentences (sharing the same sha256_hash) are strictly
    assigned to the same partition and NEVER leak across train, val, and test.
    """
    # Create synthetic records with intentional duplicate sentence hashes
    records = []
    # 30 unique sentences, each duplicated 2 or 3 times
    for i in range(30):
        h = f"hash_{i:04d}"
        label = i % 3  # Balanced labels: 0, 1, 2
        domain = "news_politics" if i < 15 else "ecommerce_reviews"
        # Duplicate each sentence 2 times
        for dup in range(2):
            records.append({
                "id": f"rec_{i}_{dup}",
                "sha256_hash": h,
                "text": f"नेपाली परीक्षण वाक्य {i} दोहोरिएको {dup}।",
                "label": label,
                "domain": domain,
            })
    
    assert len(records) == 60  # 60 records, 30 unique hashes
    
    splits = hash_grouped_stratified_split(
        records,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    
    train_hashes = set(r["sha256_hash"] for r in splits["train"])
    val_hashes = set(r["sha256_hash"] for r in splits["val"])
    test_hashes = set(r["sha256_hash"] for r in splits["test"])
    
    # 1. Zero Hash Overlap Guarantees
    assert len(train_hashes & val_hashes) == 0
    assert len(train_hashes & test_hashes) == 0
    assert len(val_hashes & test_hashes) == 0
    
    # 2. All unique sentence hashes partitioned (unanimous duplicates collapsed to 1 canonical record)
    total_split_records = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total_split_records == 30
    assert all("provenance_sources" in r for r in splits["train"])
    
    # 3. Zero leakage verification flag in metadata
    assert splits["metadata"]["zero_leakage_verified"] is True


def test_contradictory_duplicate_quarantine():
    """
    Verify Invariant B1:
    - If a hash cluster has conflicting labels without adjudication, quarantine to conflicts
    - If a hash cluster has unanimous labels, collapse to 1 canonical record and merge provenance_sources
    """
    records = [
        # Conflicting duplicate: same text/hash, differing labels (0 vs 2), not adjudicated
        {"id": "c1_a", "sha256_hash": "hash_conflict_1", "text": "विवादित वाक्य", "label": 0, "domain": "news"},
        {"id": "c1_b", "sha256_hash": "hash_conflict_1", "text": "विवादित वाक्य", "label": 2, "domain": "news"},
        # Adjudicated duplicate: differing labels but marked is_adjudicated=True
        {"id": "adj_a", "sha256_hash": "hash_adj", "text": "निर्णय भएको वाक्य", "label": 1, "domain": "news", "is_adjudicated": True},
        # Unanimous duplicates with multiple sources
        {"id": "u1_a", "sha256_hash": "hash_unan_1", "text": "सहमति वाक्य", "label": 2, "domain": "reviews", "source": "src1"},
        {"id": "u1_b", "sha256_hash": "hash_unan_1", "text": "सहमति वाक्य", "label": 2, "domain": "reviews", "source": "src2"},
    ]
    
    splits = hash_grouped_stratified_split(records, seed=42)
    
    # Conflicting records quarantined
    assert len(splits["conflicts"]) == 2
    conflict_ids = {r["id"] for r in splits["conflicts"]}
    assert conflict_ids == {"c1_a", "c1_b"}
    
    # Adjudicated and unanimous records partitioned
    partitioned_hashes = (
        {r["sha256_hash"] for r in splits["train"]}
        | {r["sha256_hash"] for r in splits["val"]}
        | {r["sha256_hash"] for r in splits["test"]}
    )
    assert "hash_conflict_1" not in partitioned_hashes
    assert "hash_adj" in partitioned_hashes
    assert "hash_unan_1" in partitioned_hashes
    
    # Check unanimous duplicate collapsed to 1 record with merged provenance
    unan_records = [r for r in splits["train"] + splits["val"] + splits["test"] if r["sha256_hash"] == "hash_unan_1"]
    assert len(unan_records) == 1
    assert "src1" in unan_records[0]["provenance_sources"]
    assert "src2" in unan_records[0]["provenance_sources"]


def test_stratification_fallback_to_label_only():
    """Verify fallback to label-only stratification when composite cell has < 2 samples."""
    records = [
        {"id": "r1", "sha256_hash": "h1", "text": "वाक्य १", "label": 0, "domain": "news"},
        {"id": "r2", "sha256_hash": "h2", "text": "वाक्य २", "label": 0, "domain": "reviews"}, # only 1 sample for (0, reviews)
        {"id": "r3", "sha256_hash": "h3", "text": "वाक्य ३", "label": 1, "domain": "news"},
        {"id": "r4", "sha256_hash": "h4", "text": "वाक्य ४", "label": 1, "domain": "news"},
    ]
    splits = hash_grouped_stratified_split(records, seed=42)
    assert splits["metadata"]["stratification_mode"] == "label_only_fallback"
    assert splits["metadata"]["zero_leakage_verified"] is True



def test_splitter_statistical_power_warning():
    """Verify that the splitter issues a warning when test class support is < 80 samples."""
    # 20 records (way below 80 samples per class)
    records = [
        {"id": f"r_{i}", "sha256_hash": f"h_{i}", "text": f"वाक्य {i}", "label": i % 3, "domain": "news"}
        for i in range(20)
    ]
    
    splits = hash_grouped_stratified_split(records, seed=42, min_test_support=80)
    warnings = splits["metadata"]["warnings"]
    assert len(warnings) > 0
    assert any("Statistical Power Warning" in w for w in warnings)
    assert any("80" in w for w in warnings)

