"""
Phase 2: Merge all annotations into final annotated dataset.

Combines:
- Validation annotations (from both annotators)
- Training annotations (from Annotator A and B separately)

Output: 04_annotated_dataset.jsonl (ready for Phase 3: Model Training)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))


class AnnotationMerger:
    """Merge annotations from multiple files into final dataset."""
    
    def __init__(self, data_dir: Path):
        """Initialize merger."""
        self.data_dir = data_dir
        self.merged_records = {}
    
    def load_and_merge(self) -> int:
        """
        Load annotations from all files and merge.
        
        Returns:
            Total merged records
        """
        
        # Files to merge
        files_to_merge = [
            self.data_dir / "03_validation_partition.jsonl",
            self.data_dir / "03_training_annotator_A.jsonl",
            self.data_dir / "03_training_annotator_B.jsonl"
        ]
        
        merged_count = 0
        
        for file_path in files_to_merge:
            if not file_path.exists():
                print(f"⚠ Skipping (not found): {file_path.name}")
                continue
            
            print(f"Processing: {file_path.name}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        record = json.loads(line)
                        
                        # Skip unannotated records
                        if record['label'] is None:
                            print(f"  ⚠ Skipping unannotated: {record['id']}")
                            continue
                        
                        # Create normalized output record
                        output_record = {
                            "id": record['id'],
                            "sentence": record['sentence'],
                            "domain": record['domain'],
                            "platform": record['platform'],
                            "source_url": record.get('source_url', ''),
                            "label": record['label'],  # 0=Negative, 1=Positive, 2=Neutral
                            "annotator_id": record['annotator_id'],
                            "partition": record.get('partition', 'unknown'),
                            "confidence": record.get('confidence'),
                            "notes": record.get('notes', ''),
                            "created_at": record.get('created_at'),
                            "annotated_at": record.get('annotated_at')
                        }
                        
                        self.merged_records[record['id']] = output_record
                        merged_count += 1
                    
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"  ✗ Error parsing line: {str(e)[:50]}")
                        continue
        
        print(f"\n✓ Merged {merged_count} annotated records")
        return merged_count
    
    def save_merged_dataset(self, output_file: Path) -> bool:
        """Save merged dataset to final file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in self.merged_records.values():
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"✓ Saved: {output_file}")
        return True
    
    def generate_statistics(self) -> dict:
        """Generate dataset statistics."""
        
        stats = {
            "total_records": len(self.merged_records),
            "by_label": defaultdict(int),
            "by_domain": defaultdict(int),
            "by_partition": defaultdict(int),
            "by_annotator": defaultdict(int),
            "by_confidence": defaultdict(int),
            "timestamp": datetime.now().isoformat()
        }
        
        label_names = {0: "Negative", 1: "Positive", 2: "Neutral"}
        
        for record in self.merged_records.values():
            label = record['label']
            domain = record['domain']
            partition = record['partition']
            annotator = record['annotator_id']
            confidence = record['confidence']
            
            stats["by_label"][label_names.get(label, f"Unknown({label})")] += 1
            stats["by_domain"][domain] += 1
            stats["by_partition"][partition] += 1
            stats["by_annotator"][annotator] += 1
            if confidence:
                stats["by_confidence"][confidence] += 1
        
        return stats


def main():
    """Execute annotation merging."""
    
    print("=" * 80)
    print(" PHASE 2: MERGE ANNOTATIONS")
    print("=" * 80)
    print()
    
    data_dir = Path("d:/ds mdoel/data")
    output_file = data_dir / "04_annotated_dataset.jsonl"
    stats_file = data_dir / "04_annotation_statistics.json"
    
    # Merge annotations
    merger = AnnotationMerger(data_dir)
    merged_count = merger.load_and_merge()
    
    if merged_count == 0:
        print("\n✗ No annotated records found.")
        print("   Please complete annotations first.")
        sys.exit(1)
    
    # Save merged dataset
    print()
    merger.save_merged_dataset(output_file)
    
    # Generate statistics
    print()
    print("=" * 80)
    print(" ANNOTATED DATASET STATISTICS")
    print("=" * 80)
    print()
    
    stats = merger.generate_statistics()
    
    print(f"Total annotations: {stats['total_records']}")
    print()
    
    print("By Label:")
    for label, count in sorted(stats['by_label'].items()):
        pct = (count / stats['total_records']) * 100
        print(f"  {label:15s}: {count:4d} ({pct:5.1f}%)")
    print()
    
    print("By Domain:")
    for domain, count in sorted(stats['by_domain'].items()):
        pct = (count / stats['total_records']) * 100
        print(f"  {domain:20s}: {count:4d} ({pct:5.1f}%)")
    print()
    
    print("By Partition:")
    for partition, count in sorted(stats['by_partition'].items()):
        pct = (count / stats['total_records']) * 100
        print(f"  {partition:15s}: {count:4d} ({pct:5.1f}%)")
    print()
    
    print("By Annotator:")
    for annotator, count in sorted(stats['by_annotator'].items()):
        pct = (count / stats['total_records']) * 100
        print(f"  {annotator:20s}: {count:4d} ({pct:5.1f}%)")
    print()
    
    if stats['by_confidence']:
        print("By Confidence:")
        for conf, count in sorted(stats['by_confidence'].items()):
            pct = (count / stats['total_records']) * 100
            print(f"  {conf:10s}: {count:4d} ({pct:5.1f}%)")
        print()
    
    # Save statistics
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Statistics saved: {stats_file}")
    print()
    
    print("=" * 80)
    print(" READY FOR PHASE 3: MODEL TRAINING")
    print("=" * 80)
    print()
    print("Next steps:")
    print(f"  1. Annotated dataset ready: {output_file}")
    print("  2. Start Phase 3: Model benchmarking")
    print("     • Level 1: TF-IDF + Logistic Regression")
    print("     • Level 2: fastText embeddings")
    print("     • Level 3: Fine-tuned mBERT")
    print("     • Level 4: Fine-tuned MuRIL")
    print()


if __name__ == "__main__":
    main()
