"""
Generate comprehensive pipeline reports and analysis.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from nepali_scraper.utils import (
    generate_domain_balance_report,
    validate_jsonl_file,
    print_report
)

def analyze_corpus():
    """Generate analysis reports for both raw and cleaned corpus."""
    
    data_dir = Path("d:/ds mdoel/data")
    
    print("=" * 80)
    print(" NEPALI SENTIMENT ANALYSIS - PIPELINE EXECUTION REPORT")
    print("=" * 80)
    print()
    
    # === RAW CORPUS ANALYSIS ===
    raw_file = data_dir / "01_raw_corpus.jsonl"
    if raw_file.exists():
        print("[1] RAW CORPUS STATISTICS")
        print("-" * 80)
        
        validation = validate_jsonl_file(raw_file)
        print(f"File: {raw_file.name}")
        print(f"Total lines: {validation['total_lines']}")
        print(f"Valid records: {validation['valid_lines']}")
        print(f"Invalid records: {validation['invalid_lines']}")
        print()
        
        report = generate_domain_balance_report(raw_file)
        
        print("Domain Distribution:")
        for domain, stats in report['domain_distribution'].items():
            pct = stats['percentage']
            count = stats['count']
            bar = "█" * int(pct / 5)
            print(f"  {domain:20s}: {count:4d} records ({pct:5.1f}%) {bar}")
        print()
        
        print("Platform Distribution:")
        for platform, stats in sorted(report['platform_distribution'].items()):
            count = stats['count']
            pct = stats['percentage']
            print(f"  {platform:20s}: {count:4d} records ({pct:5.1f}%)")
        print()
    
    # === CLEANED CORPUS ANALYSIS ===
    cleaned_file = data_dir / "02_cleaned_corpus.jsonl"
    if cleaned_file.exists():
        print("[2] CLEANED CORPUS STATISTICS")
        print("-" * 80)
        
        validation = validate_jsonl_file(cleaned_file)
        print(f"File: {cleaned_file.name}")
        print(f"Total sentences: {validation['valid_lines']}")
        print(f"Invalid records: {validation['invalid_lines']}")
        print()
        
        report = generate_domain_balance_report(cleaned_file)
        
        print("Domain Distribution (After Processing):")
        for domain, stats in report['domain_distribution'].items():
            pct = stats['percentage']
            count = stats['count']
            bar = "█" * int(pct / 5)
            print(f"  {domain:20s}: {count:4d} sentence ({pct:5.1f}%) {bar}")
        print()
        
        # Analyze sentence statistics
        print("Sentence Quality Metrics:")
        lengths = []
        with open(cleaned_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        sentence = record.get('sentence', '')
                        lengths.append(len(sentence))
                    except:
                        pass
        
        if lengths:
            lengths.sort()
            print(f"  Min length: {min(lengths)} characters")
            print(f"  Max length: {max(lengths)} characters")
            print(f"  Avg length: {sum(lengths) / len(lengths):.1f} characters")
            print(f"  Median length: {lengths[len(lengths)//2]} characters")
            print()
    
    # === PIPELINE EFFICIENCY ===
    if raw_file.exists() and cleaned_file.exists():
        print("[3] PIPELINE EFFICIENCY METRICS")
        print("-" * 80)
        
        raw_count = validate_jsonl_file(raw_file)['valid_lines']
        cleaned_count = validate_jsonl_file(cleaned_file)['valid_lines']
        
        compression = cleaned_count / raw_count if raw_count > 0 else 0
        data_loss = (1 - compression) * 100 if raw_count > 0 else 0
        
        print(f"Raw records input: {raw_count}")
        print(f"Cleaned sentences output: {cleaned_count}")
        print(f"Compression ratio: {compression:.2f} (sentences per record)")
        print(f"Data loss rate: {data_loss:.1f}%")
        print()
        
        # Growth/reduction status
        if compression < 0.5:
            loss_status = "⚠ HIGH (>50%)"
        elif compression < 0.6:
            loss_status = "⚠ MODERATE (40-60%)"
        else:
            loss_status = "✓ NORMAL (<40%)"
        
        print(f"Loss assessment: {loss_status}")
        print()
    
    # === NEXT PHASE GUIDANCE ===
    print("[4] NEXT PHASE: HUMAN ANNOTATION")
    print("-" * 80)
    
    if cleaned_file.exists() and cleaned_count > 0:
        val_size = int(cleaned_count * 0.20)
        train_size = cleaned_count - val_size
        
        print(f"Dataset-ready corpus: {cleaned_count} sentences")
        print()
        print("Recommended split for Phase 2:")
        print(f"  • Validation set (20%): {val_size} sentences")
        print(f"    → Assign to BOTH annotators (inter-annotator agreement)")
        print()
        print(f"  • Training set (80%): {train_size} sentences")
        print(f"    → Split evenly between Annotator A & B")
        print(f"    → Annotator A: {train_size // 2} sentences")
        print(f"    → Annotator B: {train_size // 2} sentences")
        print()
        print("Expected Cohen's Kappa target: κ ≥ 0.60 (substantial agreement)")
        print()
    
    print("=" * 80)
    print()

if __name__ == "__main__":
    analyze_corpus()
