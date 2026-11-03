"""
Data intake, provenance validation, and deduplication engine for NepSent-Engine.

Provides:
- DatasetIngestor: Validates raw data provenance metadata (text, domain, source, timestamp, license)
- Text quality filtration (NFC normalization, length check, Devanagari ratio >= 0.70, sha256_hash)
- Exact deduplication via SHA-256 sentence hashing
- Near-duplicate detection via character n-gram Jaccard similarity (threshold >= 0.85)
- Hindi and foreign-Devanagari contamination detection using high-frequency Hindi-exclusive function words
"""

import re
import csv
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Set, Optional, Tuple, Iterable, Any

from nepali_scraper.config import PROJECT_ROOT, DOMAIN_TARGETS
from nepali_scraper.processors import (
    NepaliTextProcessor,
    compute_sentence_hash,
    devanagari_ratio,
)

logger = logging.getLogger(__name__)

# Canonical domain taxonomy
CANONICAL_DOMAINS = {
    "news": "news_politics",
    "news_politics": "news_politics",
    "politics": "news_politics",
    "reviews": "ecommerce_reviews",
    "ecommerce": "ecommerce_reviews",
    "ecommerce_reviews": "ecommerce_reviews",
    "social": "social_media",
    "social_media": "social_media",
    "twitter": "social_media",
    "reddit": "social_media",
}

# High-frequency Hindi-exclusive function words and auxiliaries
# These words do not occur in authentic standard Nepali text
HINDI_EXCLUSIVE_MARKERS = {
    "है", "हैं", "था", "थी", "थे", "क्योंकि", "लेकिन", "नहीं", "किया", "कियागया",
    "करते", "करती", "रहा", "रही", "रहे", "होता", "होती", "होते", "मुझे", "हमे",
    "आपको", "उनको", "इसका", "उसका", "बहुत"
}

# Tokenization regex for Devanagari words excluding sentence delimiters (U+0964 Purna Viram, U+0965 Deergha Viram)
WORD_TOKEN_PATTERN = re.compile(r'[\u0900-\u0963\u0966-\u097F\w]+')


def detect_hindi_contamination(text: str) -> Tuple[bool, List[str]]:
    """
    Detect whether Devanagari text contains Hindi-exclusive function words.
    
    Args:
        text: Input string
        
    Returns:
        (is_contaminated, list_of_detected_markers)
    """
    if not text:
        return False, []
    
    tokens = set(WORD_TOKEN_PATTERN.findall(text))
    detected = [word for word in tokens if word in HINDI_EXCLUSIVE_MARKERS]
    return len(detected) > 0, detected


def char_ngram_jaccard(text1: str, text2: str, n: int = 3) -> float:
    """
    Compute Jaccard similarity of character n-grams between two strings.
    
    Args:
        text1: First string
        text2: Second string
        n: N-gram length (default 3)
        
    Returns:
        Float between 0.0 and 1.0 representing Jaccard similarity
    """
    if not text1 or not text2:
        return 1.0 if text1 == text2 else 0.0
    
    # If strings are identical
    if text1 == text2:
        return 1.0
    
    # If strings are shorter than n
    if len(text1) < n or len(text2) < n:
        return 1.0 if text1 == text2 else 0.0
    
    ngrams1 = set(text1[i:i + n] for i in range(len(text1) - n + 1))
    ngrams2 = set(text2[i:i + n] for i in range(len(text2) - n + 1))
    
    union = ngrams1 | ngrams2
    if not union:
        return 0.0
    
    intersection = ngrams1 & ngrams2
    return len(intersection) / len(union)


class DatasetIngestor:
    """
    Clean, extensible dataset ingestion and provenance validator.
    
    Validates provenance, passes text through normalization and length constraints,
    enforces minimum Devanagari ratio (>= 0.70), flags Hindi contamination,
    and removes exact duplicates (SHA-256) and near-duplicates (Jaccard >= 0.85).
    """
    
    def __init__(
        self,
        min_devanagari_ratio: float = 0.70,
        min_length: int = 15,
        max_length: int = 350,
        near_duplicate_threshold: float = 0.85,
        ngram_size: int = 3,
        drop_hindi_contaminated: bool = True,
    ):
        self.min_devanagari_ratio = min_devanagari_ratio
        self.min_length = min_length
        self.max_length = max_length
        self.near_duplicate_threshold = near_duplicate_threshold
        self.ngram_size = ngram_size
        self.drop_hindi_contaminated = drop_hindi_contaminated
        
        self.processor = NepaliTextProcessor(config={
            "min_length": min_length,
            "max_length": max_length,
            "min_devanagari_ratio": min_devanagari_ratio,
        })
        
        # Deduplication registries
        self.seen_hashes: Set[str] = set()
        self.accepted_records: List[Dict[str, Any]] = []
        self._accepted_ngram_cache: List[Tuple[str, Set[str]]] = []  # (hash, ngrams)
        
        # Audit statistics
        self.stats = {
            "total_raw": 0,
            "accepted": 0,
            "rejected_missing_metadata": 0,
            "rejected_invalid_domain": 0,
            "rejected_length": 0,
            "rejected_devanagari_ratio": 0,
            "rejected_exact_duplicate": 0,
            "rejected_near_duplicate": 0,
            "flagged_hindi_contamination": 0,
        }
    
    def canonicalize_domain(self, domain_raw: str) -> Optional[str]:
        """Convert domain variations to canonical domain identifier."""
        if not domain_raw:
            return None
        return CANONICAL_DOMAINS.get(domain_raw.strip().lower())
    
    def _is_near_duplicate(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if text is near-duplicate with any previously accepted record.
        
        Args:
            text: NFC normalized sentence
            
        Returns:
            (is_near_dup, matched_hash_or_None)
        """
        if len(text) < self.ngram_size or not self._accepted_ngram_cache:
            return False, None
        
        ngrams = set(text[i:i + self.ngram_size] for i in range(len(text) - self.ngram_size + 1))
        if not ngrams:
            return False, None
        
        for prev_hash, prev_ngrams in self._accepted_ngram_cache:
            union = ngrams | prev_ngrams
            if not union:
                continue
            similarity = len(ngrams & prev_ngrams) / len(union)
            if similarity >= self.near_duplicate_threshold:
                return True, prev_hash
        
        return False, None
    
    def process_and_validate_record(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Validate, clean, normalize, and deduplicate a single raw record.
        
        Args:
            raw: Dictionary containing raw record fields
            
        Returns:
            Validated record dictionary or None if rejected
        """
        self.stats["total_raw"] += 1
        
        # 1. Validate mandatory provenance metadata
        text = raw.get("text", "")
        if not isinstance(text, str) or not text.strip():
            self.stats["rejected_missing_metadata"] += 1
            return None
        
        domain_raw = raw.get("domain", "")
        canonical_domain = self.canonicalize_domain(domain_raw)
        if not canonical_domain:
            self.stats["rejected_invalid_domain"] += 1
            return None
        
        source = raw.get("source", "").strip() if raw.get("source") else "unspecified"
        license_str = raw.get("license", "").strip() if raw.get("license") else "unspecified"
        
        timestamp = raw.get("collection_timestamp")
        if not timestamp:
            timestamp = datetime.now(timezone.utc).isoformat()
        
        # 2. Text cleaning & Unicode NFC normalization
        cleaned_text = self.processor.clean_and_normalize(text)
        
        # 3. Length validation
        if not (self.min_length <= len(cleaned_text) <= self.max_length):
            self.stats["rejected_length"] += 1
            return None
        
        # 4. Devanagari ratio validation (>= 0.70)
        ratio = devanagari_ratio(cleaned_text)
        if ratio < self.min_devanagari_ratio:
            self.stats["rejected_devanagari_ratio"] += 1
            return None
        
        # 5. Hindi & foreign Devanagari contamination check
        is_contaminated, hindi_words = detect_hindi_contamination(cleaned_text)
        if is_contaminated:
            self.stats["flagged_hindi_contamination"] += 1
            if self.drop_hindi_contaminated:
                return None
        
        # 6. Compute exact SHA-256 sentence hash
        sentence_hash = compute_sentence_hash(cleaned_text)
        if sentence_hash in self.seen_hashes:
            self.stats["rejected_exact_duplicate"] += 1
            return None
        
        # 7. Check near-duplicate (Jaccard similarity >= 0.85)
        is_near_dup, matched_hash = self._is_near_duplicate(cleaned_text)
        if is_near_dup:
            self.stats["rejected_near_duplicate"] += 1
            return None
        
        # Record passes all validation checks
        record_id = raw.get("id") or f"nep_{sentence_hash[:12]}"
        
        validated_record = {
            "id": record_id,
            "text": cleaned_text,
            "domain": canonical_domain,
            "source": source,
            "collection_timestamp": timestamp,
            "license": license_str,
            "sha256_hash": sentence_hash,
            "devanagari_ratio": round(ratio, 4),
            "char_length": len(cleaned_text),
            "provenance_status": raw.get("provenance_status", "REAL_COLLECTED"),
            "schema_version": "2.0-ordinal",
            "is_contaminated": is_contaminated,
            "hindi_markers": hindi_words,
        }
        
        # Preserve ground truth label if already annotated
        if "label" in raw and raw["label"] is not None:
            validated_record["label"] = int(raw["label"])
        
        # Register hash and cache n-grams for subsequent near-duplicate checks
        self.seen_hashes.add(sentence_hash)
        self.accepted_records.append(validated_record)
        
        ngrams = set(cleaned_text[i:i + self.ngram_size] for i in range(len(cleaned_text) - self.ngram_size + 1))
        self._accepted_ngram_cache.append((sentence_hash, ngrams))
        self.stats["accepted"] += 1
        
        return validated_record
    
    def ingest_records(self, records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ingest and validate an iterable of raw dictionary records."""
        results = []
        for r in records:
            res = self.process_and_validate_record(r)
            if res:
                results.append(res)
        return results
    
    def ingest_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Ingest raw records from a JSONL file."""
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        record = json.loads(line_str)
                        records.append(record)
                    except json.JSONDecodeError:
                        logger.warning(f"Corrupt JSON line skipped in {path}")
        return self.ingest_records(records)
    
    def ingest_csv(
        self,
        file_path: Path,
        text_col: str = "text",
        domain_col: str = "domain",
        source_col: str = "source",
        license_col: str = "license",
        label_col: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Ingest raw records from a CSV file."""
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        
        records = []
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec = {
                    "text": row.get(text_col, ""),
                    "domain": row.get(domain_col, ""),
                    "source": row.get(source_col, "csv_import"),
                    "license": row.get(license_col, "unspecified"),
                    "collection_timestamp": datetime.now(timezone.utc).isoformat(),
                }
                if label_col and label_col in row and row[label_col] != "":
                    try:
                        rec["label"] = int(row[label_col])
                    except ValueError:
                        pass
                records.append(rec)
        return self.ingest_records(records)
    
    def save_accepted_jsonl(self, output_path: Path) -> Path:
        """Write all accepted records to a JSONL file."""
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for record in self.accepted_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path
    
    def get_summary(self) -> Dict[str, Any]:
        """Return audit summary of ingestion operation."""
        return {
            "stats": dict(self.stats),
            "unique_hashes_count": len(self.seen_hashes),
            "accepted_count": len(self.accepted_records),
        }
