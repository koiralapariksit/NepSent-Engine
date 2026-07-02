"""
Run text processing pipeline on synthetic dataset.
Demonstrates full data flow: raw → cleaned → processed.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from nepali_scraper.processors import NepaliTextProcessor, is_valid_nepali_text
from datetime import datetime

def process_synthetic_dataset():
    """Process raw corpus through text processing pipeline."""
    
    data_dir = Path("d:/ds mdoel/data")
    input_file = data_dir / "01_raw_corpus.jsonl"
    output_file = data_dir / "02_cleaned_corpus.jsonl"
    
    if not input_file.exists():
        print(f"✗ Input file not found: {input_file}")
        return False
    
    print("[*] Running text processing pipeline...")
    print(f"    Input: {input_file}")
    
    processor = NepaliTextProcessor({
        "min_length": 15,
        "max_length": 350
    })
    
    sentence_count = 0
    record_count = 0
    discarded_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as inf:
        with open(output_file, 'w', encoding='utf-8') as outf:
            for line in inf:
                if not line.strip():
                    continue
                
                try:
                    record = json.loads(line)
                    record_count += 1
                    
                    # Extract text based on domain
                    domain = record.get('domain', 'unknown')
                    
                    if domain == "news_politics":
                        text = record.get('content', '') or record.get('headline', '')
                    elif domain == "ecommerce_reviews":
                        text = record.get('review_text', '')
                    elif domain == "social_media":
                        text = record.get('text', '')
                    else:
                        text = record.get('content', '') or record.get('text', '')
                    
                    if not text:
                        discarded_count += 1
                        continue
                    
                    # Validate Nepali content
                    if not is_valid_nepali_text(text):
                        discarded_count += 1
                        continue
                    
                    # Process into sentences
                    sentences = processor.segment_sentences(text)
                    
                    for sentence in sentences:
                        cleaned_record = {
                            "domain": domain,
                            "platform": record.get('platform', 'unknown'),
                            "sentence": sentence,
                            "source_url": record.get('url', ''),
                            "processed_at": datetime.now().isoformat()
                        }
                        outf.write(json.dumps(cleaned_record, ensure_ascii=False) + '\n')
                        sentence_count += 1
                
                except json.JSONDecodeError as e:
                    print(f"✗ JSON error on line: {str(e)[:50]}")
                    continue
    
    print(f"\n✓ PIPELINE RESULTS:")
    print(f"  - Input records processed: {record_count}")
    print(f"  - Sentences extracted: {sentence_count}")
    print(f"  - Records discarded: {discarded_count}")
    print(f"  - Compression ratio: {sentence_count / record_count:.2f} sentences/record")
    print(f"  - Output file: {output_file}")
    print()
    
    return True

if __name__ == "__main__":
    process_synthetic_dataset()
