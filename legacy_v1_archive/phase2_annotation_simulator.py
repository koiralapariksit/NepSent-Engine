"""
Phase 2: Simulate annotations for demonstration.
In production, human annotators would manually complete the annotation files.

This script demonstrates:
- How to fill in validation and training annotations
- Expected output format
- How to run quality analysis
"""

import json
import random
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from phase2_annotation_guidelines import LABEL_SCHEMA


class AnnotationSimulator:
    """
    Simulate human annotations for demo purposes.
    
    In production, replace this with actual human annotation.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize simulator with optional seed for reproducibility."""
        random.seed(seed)
    
    def simulate_label(self, sentence: str, domain: str) -> int:
        """
        Simulate annotation logic (heuristic-based for demo).
        
        In production, this is replaced by actual human judgment.
        """
        sentence_lower = sentence.lower()
        
        # Simple rule-based heuristic
        negative_words = ['खराब', 'नराम्रो', 'असन्तुष्ट', 'गुन्साउनु', 'दु:ख', 'समस्या', 'संकट']
        positive_words = ['राम्रो', 'सुपरको', 'खुस', 'सन्तुष्ट', 'तारिफ', 'प्रभाव', 'गर्विलो']
        
        neg_count = sum(1 for word in negative_words if word in sentence_lower)
        pos_count = sum(1 for word in positive_words if word in sentence_lower)
        
        if domain == "news_politics":
            # News more likely to be neutral
            if neg_count > 0 and pos_count == 0:
                return 0
            elif pos_count > 0 and neg_count == 0:
                return 1
            else:
                return random.choice([0, 1, 2]) if random.random() < 0.7 else 2
        
        elif domain == "ecommerce_reviews":
            # Reviews have strong sentiment
            if neg_count > pos_count:
                return 0
            elif pos_count > neg_count:
                return 1
            else:
                return 1 if random.random() < 0.6 else 0
        
        else:  # social_media
            # Social media mixed
            if neg_count > 0 and pos_count == 0:
                return 0
            elif pos_count > 0 and neg_count == 0:
                return 1
            else:
                return random.randint(0, 2)
    
    def annotate_file(self, input_file: Path, annotator_id: str) -> int:
        """
        Simulate annotation of a file.
        
        Args:
            input_file: Path to annotation file (e.g., 03_validation_partition.jsonl)
            annotator_id: Identifier for this annotator (must match file)
            
        Returns:
            Number of annotations added
        """
        if not input_file.exists():
            print(f"✗ File not found: {input_file}")
            return 0
        
        records = []
        annotation_count = 0
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    record = json.loads(line)
                    
                    # Only process records for this annotator
                    if record['annotator_id'] != annotator_id:
                        records.append(record)
                        continue
                    
                    # Skip if already annotated (has non-None label)
                    if record['label'] is not None:
                        records.append(record)
                        continue
                    
                    # Simulate annotation
                    label = self.simulate_label(record['sentence'], record['domain'])
                    confidence = random.choice(['HIGH', 'MEDIUM', 'LOW'])
                    
                    record['label'] = label
                    record['confidence'] = confidence
                    record['annotated_at'] = datetime.now().isoformat()
                    
                    # Add occasional notes for low-confidence annotations
                    if confidence == 'LOW':
                        record['notes'] = "Uncertain - multiple interpretations possible"
                    
                    records.append(record)
                    annotation_count += 1
                
                except (json.JSONDecodeError, KeyError):
                    pass
        
        # Write back
        with open(input_file, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        return annotation_count


def main():
    """Simulate both annotators completing validation set."""
    
    print("=" * 80)
    print(" PHASE 2: ANNOTATION SIMULATION (for demonstration)")
    print("=" * 80)
    print()
    print("⚠ NOTE: This is a SIMULATION for demonstration.")
    print("   In production, human annotators complete the annotation files.")
    print()
    
    data_dir = Path("d:/ds mdoel/data")
    val_file = data_dir / "03_validation_partition.jsonl"
    
    if not val_file.exists():
        print(f"✗ Validation file not found: {val_file}")
        print("  Please run: python phase2_data_split.py")
        sys.exit(1)
    
    simulator = AnnotationSimulator(seed=42)
    
    # Simulate Annotator A
    print("Simulating Annotator A annotations...")
    count_A = simulator.annotate_file(val_file, "annotator_A")
    print(f"✓ Annotator A: {count_A} sentences annotated")
    print()
    
    # Reset seed for different results (Annotator B)
    simulator = AnnotationSimulator(seed=43)
    
    # Simulate Annotator B
    print("Simulating Annotator B annotations...")
    count_B = simulator.annotate_file(val_file, "annotator_B")
    print(f"✓ Annotator B: {count_B} sentences annotated")
    print()
    
    # Simulate training set annotations
    print("Simulating training set annotations...")
    
    train_A_file = data_dir / "03_training_annotator_A.jsonl"
    train_B_file = data_dir / "03_training_annotator_B.jsonl"
    
    if train_A_file.exists():
        simulator = AnnotationSimulator(seed=44)
        count_train_A = simulator.annotate_file(train_A_file, "annotator_A")
        print(f"✓ Training set (Annotator A): {count_train_A} sentences annotated")
    
    if train_B_file.exists():
        simulator = AnnotationSimulator(seed=45)
        count_train_B = simulator.annotate_file(train_B_file, "annotator_B")
        print(f"✓ Training set (Annotator B): {count_train_B} sentences annotated")
    
    print()
    print("=" * 80)
    print(" SIMULATION COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Calculate Cohen's Kappa: python phase2_kappa_calculation.py")
    print("  2. If κ ≥ 0.60: Merge annotations: python phase2_merge_annotations.py")
    print()


if __name__ == "__main__":
    main()
