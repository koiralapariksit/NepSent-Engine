"""
Phase 2: Dataset partitioning and annotation workflow setup.
Splits cleaned corpus into validation (both annotators) and training (split A/B).
"""

import json
import sys
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from phase2_annotation_guidelines import LABEL_SCHEMA


class AnnotationDatasetSplitter:
    """
    Split cleaned corpus into validation and training sets for dual-annotator workflow.
    
    Distribution:
    - Validation (20%): Both annotators label same sentences (for inter-annotator agreement)
    - Training (80%): Split evenly between Annotator A and B
    """
    
    def __init__(self, cleaned_corpus_file: Path):
        """
        Initialize splitter.
        
        Args:
            cleaned_corpus_file: Path to 02_cleaned_corpus.jsonl
        """
        self.cleaned_file = cleaned_corpus_file
        self.records = []
        self.load_corpus()
    
    def load_corpus(self):
        """Load cleaned corpus."""
        if not self.cleaned_file.exists():
            raise FileNotFoundError(f"Cleaned corpus not found: {self.cleaned_file}")
        
        with open(self.cleaned_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    record['_id'] = idx  # Unique identifier
                    self.records.append(record)
                except json.JSONDecodeError:
                    pass
        
        print(f"[*] Loaded {len(self.records)} sentences from cleaned corpus")
    
    def split_dataset(self, val_ratio: float = 0.20) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split dataset into validation and training sets.
        
        Args:
            val_ratio: Validation set ratio (default 0.20 = 20%)
            
        Returns:
            (validation_set, training_set_A, training_set_B)
        """
        total = len(self.records)
        val_size = max(1, int(total * val_ratio))
        train_size = total - val_size
        train_size_per_annotator = train_size // 2
        
        # Shuffle for random split
        shuffled = self.records.copy()
        random.shuffle(shuffled)
        
        # Validation set (same for both annotators)
        validation_set = shuffled[:val_size]
        
        # Training set (split between A and B)
        remaining = shuffled[val_size:]
        training_set_A = remaining[:train_size_per_annotator]
        training_set_B = remaining[train_size_per_annotator:]
        
        print(f"\n[SPLIT RESULTS]")
        print(f"  Total sentences: {total}")
        print(f"  Validation (both annotators): {len(validation_set)} sentences")
        print(f"    → Annotator A records: {len(validation_set)}")
        print(f"    → Annotator B records: {len(validation_set)}")
        print(f"  Training (Annotator A): {len(training_set_A)} sentences")
        print(f"  Training (Annotator B): {len(training_set_B)} sentences")
        print(f"  ───────────")
        print(f"  Total assigned: {len(validation_set) * 2 + len(training_set_A) + len(training_set_B)} annotations")
        
        return validation_set, training_set_A, training_set_B
    
    def save_annotation_files(self, output_dir: Path, val_set: List[Dict], 
                             train_A: List[Dict], train_B: List[Dict]):
        """
        Save annotation files for each annotator.
        
        Files created:
        - 03_validation_partition.jsonl (both annotators)
        - 03_training_annotator_A.jsonl
        - 03_training_annotator_B.jsonl
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # === VALIDATION SET ===
        # Both annotators get the same validation sentences
        val_file = output_dir / "03_validation_partition.jsonl"
        
        with open(val_file, 'w', encoding='utf-8') as f:
            # Records for Annotator A
            for sentence in val_set:
                record_A = {
                    "id": f"{sentence['_id']}_A",
                    "sentence": sentence['sentence'],
                    "domain": sentence['domain'],
                    "platform": sentence['platform'],
                    "source_url": sentence.get('source_url', ''),
                    "annotator_id": "annotator_A",
                    "partition": "validation",
                    "label": None,  # To be filled by annotator
                    "confidence": None,
                    "notes": "",
                    "created_at": datetime.now().isoformat()
                }
                f.write(json.dumps(record_A, ensure_ascii=False) + '\n')
            
            # Records for Annotator B (same sentences)
            for sentence in val_set:
                record_B = {
                    "id": f"{sentence['_id']}_B",
                    "sentence": sentence['sentence'],
                    "domain": sentence['domain'],
                    "platform": sentence['platform'],
                    "source_url": sentence.get('source_url', ''),
                    "annotator_id": "annotator_B",
                    "partition": "validation",
                    "label": None,  # To be filled by annotator
                    "confidence": None,
                    "notes": "",
                    "created_at": datetime.now().isoformat()
                }
                f.write(json.dumps(record_B, ensure_ascii=False) + '\n')
        
        print(f"✓ Saved validation set: {val_file}")
        print(f"  → {len(val_set)} sentences for Annotator A")
        print(f"  → {len(val_set)} sentences for Annotator B")
        
        # === TRAINING SET A ===
        train_A_file = output_dir / "03_training_annotator_A.jsonl"
        
        with open(train_A_file, 'w', encoding='utf-8') as f:
            for sentence in train_A:
                record = {
                    "id": f"{sentence['_id']}_A",
                    "sentence": sentence['sentence'],
                    "domain": sentence['domain'],
                    "platform": sentence['platform'],
                    "source_url": sentence.get('source_url', ''),
                    "annotator_id": "annotator_A",
                    "partition": "training",
                    "label": None,
                    "confidence": None,
                    "notes": "",
                    "created_at": datetime.now().isoformat()
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"✓ Saved training set: {train_A_file}")
        print(f"  → {len(train_A)} sentences for Annotator A")
        
        # === TRAINING SET B ===
        train_B_file = output_dir / "03_training_annotator_B.jsonl"
        
        with open(train_B_file, 'w', encoding='utf-8') as f:
            for sentence in train_B:
                record = {
                    "id": f"{sentence['_id']}_B",
                    "sentence": sentence['sentence'],
                    "domain": sentence['domain'],
                    "platform": sentence['platform'],
                    "source_url": sentence.get('source_url', ''),
                    "annotator_id": "annotator_B",
                    "partition": "training",
                    "label": None,
                    "confidence": None,
                    "notes": "",
                    "created_at": datetime.now().isoformat()
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"✓ Saved training set: {train_B_file}")
        print(f"  → {len(train_B)} sentences for Annotator B")


def create_annotation_template():
    """Create an empty annotation template for reference."""
    template = {
        "id": "sentence_001_A",
        "sentence": "नेपालको अर्थनीति यस वर्ष बढदारी गरिरहेको छ।",
        "domain": "news_politics",
        "platform": "ratopati",
        "source_url": "https://ratopati.com/...",
        "annotator_id": "annotator_A",
        "partition": "validation",  # or "training"
        "label": 1,  # TO BE FILLED: 0=Negative, 1=Positive, 2=Neutral
        "confidence": "HIGH",  # TO BE FILLED: HIGH / MEDIUM / LOW
        "notes": "Sample reasoning if uncertain",  # Optional notes
        "created_at": "2026-06-20T12:00:00",
        "annotated_at": None  # TO BE FILLED with completion timestamp
    }
    
    return template


def main():
    """Execute Phase 2 data split."""
    
    print("=" * 80)
    print(" PHASE 2: ANNOTATION DATASET SETUP")
    print("=" * 80)
    print()
    
    # Paths
    cleaned_file = Path("d:/ds mdoel/data/02_cleaned_corpus.jsonl")
    data_dir = Path("d:/ds mdoel/data")
    
    if not cleaned_file.exists():
        print(f"✗ Cleaned corpus not found: {cleaned_file}")
        print("  Please run: python process_pipeline.py")
        return False
    
    # Create splitter
    splitter = AnnotationDatasetSplitter(cleaned_file)
    
    # Split dataset
    val_set, train_A, train_B = splitter.split_dataset(val_ratio=0.20)
    
    # Save annotation files
    splitter.save_annotation_files(data_dir, val_set, train_A, train_B)
    
    print()
    print("=" * 80)
    print(" INTER-ANNOTATOR AGREEMENT CALIBRATION")
    print("=" * 80)
    print()
    print("Validation Set Instructions:")
    print("  1. Both annotators label the SAME 18 sentences independently")
    print("  2. Save completed annotations")
    print("  3. Run: python phase2_kappa_calculation.py")
    print("  4. Check Cohen's Kappa (κ) score")
    print()
    print("Target Cohen's Kappa:")
    print("  κ ≥ 0.60  → SUBSTANTIAL agreement ✓ (ready for publication)")
    print("  κ < 0.60  → Run consensus mediation & re-label")
    print()
    
    # Show template
    print("=" * 80)
    print(" ANNOTATION TEMPLATE (JSON FORMAT)")
    print("=" * 80)
    template = create_annotation_template()
    print(json.dumps(template, ensure_ascii=False, indent=2))
    print()
    print("Fields to complete:")
    print("  • label (required): 0=Negative, 1=Positive, 2=Neutral")
    print("  • confidence (optional): HIGH / MEDIUM / LOW")
    print("  • notes (optional): Reasoning for uncertain cases")
    print("  • annotated_at: Completion timestamp")
    print()
    print("=" * 80)
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
