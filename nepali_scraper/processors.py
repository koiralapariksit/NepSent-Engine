"""
Nepali text processing pipeline.
Implements Unicode NFC normalization, zero-width character handling,
Devanagari script validation, sentence boundary segmentation, and SHA-256 hashing.
"""

import re
import unicodedata
import hashlib
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Zero-width & non-semantic characters to strip:
# \u200B: Zero-width space
# \uFEFF: Zero-width no-break space / BOM
# \u200E: Left-to-right mark (LRM)
# \u200F: Right-to-left mark (RLM)
# \u00AD: Soft hyphen
# \u2060: Word joiner
NON_SEMANTIC_ZW_PATTERN = re.compile(r'[\u200B\uFEFF\u200E\u200F\u00AD\u2060]')

# Sentence delimiters: Purna Viram (।), Deergha Viram (॥), Question (?), Exclamation (!)
SENTENCE_TERMINATOR_PATTERN = re.compile(r'([।?!॥]+)')

# Whitespace normalization pattern (includes ASCII whitespace, non-breaking space, thin/em/en spaces)
WHITESPACE_PATTERN = re.compile(r'[\t\r\f\v\u00A0\u2000-\u200A\s]+')

# High-frequency Hindi-exclusive function words and auxiliaries
HINDI_EXCLUSIVE_MARKERS = {
    "है", "हैं", "था", "थी", "थे", "क्योंकि", "लेकिन", "नहीं", "किया", "कियागया",
    "करते", "करती", "रहा", "रही", "रहे", "होता", "होती", "होते", "मुझे", "हमे",
    "आपको", "उनको", "इसका", "उसका", "बहुत"
}

# Common Hindi loanwords used colloquially in Nepali
COLLOQUIAL_LOANWORDS = {"लेकिन", "बहुत"}

# Authentic Nepali verbal inflections and grammatical auxiliaries
NEPALI_VERBAL_INFLECTIONS = {
    "छ", "छन्", "थियो", "थिइन्", "थिई", "भयो", "गर्छ", "गर्छन्", "छैन", "छैनन्", "होइन",
    "हुनेछ", "गरेको", "गरेका", "भने", "गर्नु", "थिए", "भनेर", "रहेको", "रहेका", "हुन्"
}

WORD_TOKEN_PATTERN = re.compile(r'[\u0900-\u0963\u0966-\u097F\w]+')


def clean_zero_width_characters(text: str) -> str:
    """
    Clean zero-width characters while preserving valid conjunct joiners.
    
    Removes invisible non-semantic zero-width spaces (ZWSP, BOM, LRM, RLM).
    Collapses multiple consecutive ZWJ (\\u200D) and ZWNJ (\\u200C) characters
    which are required for Devanagari conjuncts and eyelash-ra.
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    # Remove non-semantic invisible characters
    text = NON_SEMANTIC_ZW_PATTERN.sub('', text)
    # Collapse duplicate ZWJ / ZWNJ
    text = re.sub(r'\u200D+', '\u200D', text)
    text = re.sub(r'\u200C+', '\u200C', text)
    return text


def normalize_nepali_unicode(text: str) -> str:
    """
    Apply Unicode NFC normalization, zero-width character handling, and whitespace collapsing.
    
    Applies Unicode Form C (NFC) canonical composition, strips non-semantic invisible
    zero-width characters (\\u200B, \\uFEFF, \\u200E, \\u200F, \\u00AD, \\u2060), preserves
    semantic Devanagari joiners \\u200C (ZWNJ) and \\u200D (ZWJ) needed for conjunct ligatures
    and eyelash-ra, and collapses arbitrary whitespace sequences into a single ASCII space.
    
    Args:
        text: Raw input string
        
    Returns:
        Canonical NFC-normalized string
    """
    if not text:
        return ""
    # Step 1: Unicode NFC normalization
    text = unicodedata.normalize('NFC', text)
    # Step 2: Clean non-semantic zero-width characters while preserving ZWJ/ZWNJ
    text = clean_zero_width_characters(text)
    # Step 3: Collapse whitespace
    text = WHITESPACE_PATTERN.sub(' ', text).strip()
    return text


def compute_hindi_contamination_score(text: str) -> float:
    """
    Compute graded Hindi contamination confidence score from 0.0 to 1.0.
    
    Distinguishes genuine Hindi grammatical contamination from benign loanwords
    (e.g., 'लेकिन', 'बहुत') when authentic Nepali verbal inflections are present.
    
    Args:
        text: Input string
        
    Returns:
        Float between 0.0 and 1.0 representing contamination confidence
    """
    if not text:
        return 0.0
    tokens = set(WORD_TOKEN_PATTERN.findall(text))
    if not tokens:
        return 0.0
        
    detected_hindi = [w for w in tokens if w in HINDI_EXCLUSIVE_MARKERS]
    if not detected_hindi:
        return 0.0
        
    detected_nepali = [w for w in tokens if w in NEPALI_VERBAL_INFLECTIONS]
    
    # If detected Hindi markers are exclusively colloquial loanwords and authentic Nepali inflections exist
    only_loanwords = all(w in COLLOQUIAL_LOANWORDS for w in detected_hindi)
    if only_loanwords and len(detected_nepali) > 0:
        # Graded low confidence (<= 0.20), not considered true grammatical contamination
        return round(min(0.20, len(detected_hindi) / (len(tokens) + 1e-5)), 4)
        
    if not detected_nepali:
        # Strong Hindi markers with zero Nepali verbal inflections
        score = min(1.0, 0.4 + 0.2 * len(detected_hindi))
        return round(score, 4)
        
    # Both Hindi markers and Nepali markers present: graded ratio
    ratio = len(detected_hindi) / (len(detected_hindi) + len(detected_nepali))
    return round(float(ratio), 4)


def compute_sentence_hash(text: str) -> str:
    """
    Compute deterministic SHA-256 hash of canonical NFC-normalized text.
    
    Used for leakage-free group-based train/test splitting and deduplication.
    
    Args:
        text: Input sentence
        
    Returns:
        Hexadecimal SHA-256 hash string
    """
    if not text:
        return hashlib.sha256(b"").hexdigest()
    canonical = normalize_nepali_unicode(text)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def devanagari_ratio(text: str) -> float:
    """
    Compute the proportion of Devanagari characters relative to total non-whitespace characters.
    
    Args:
        text: Input string
        
    Returns:
        Float between 0.0 and 1.0 representing Devanagari ratio
    """
    if not text:
        return 0.0
    non_ws = WHITESPACE_PATTERN.sub('', text)
    if not non_ws:
        return 0.0
    dev_chars = len(re.findall(r'[\u0900-\u097F]', non_ws))
    return dev_chars / len(non_ws)


def is_valid_nepali_text(text: str, min_ratio: float = 0.30) -> bool:
    """
    Validate that text contains a sufficient proportion of Devanagari script.
    
    Args:
        text: Input string
        min_ratio: Minimum ratio of Devanagari characters (default: 0.30 to allow code-mixing)
        
    Returns:
        True if text satisfies minimum Devanagari ratio
    """
    return devanagari_ratio(text) >= min_ratio


class NepaliTextProcessor:
    """
    Robust Devanagari text processor for web-scraped content and research datasets.
    
    Core Responsibilities:
    1. Unicode Form C (NFC) normalization.
    2. Zero-width character handling (systematic ZWSP/BOM stripping, ZWJ/ZWNJ preservation).
    3. Sentence boundary segmentation supporting Purna Viram (।), double danda (॥),
       question marks, exclamation marks, and newlines without dropping trailing sentences.
    4. Inclusive boundary length validation (min_length <= len <= max_length).
    5. Devanagari script ratio check to support clean code-mixing analysis.
    6. SHA-256 sentence hashing on canonical text for leak-free deduplication.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize processor with configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        if config is None:
            config = {}
        
        self.min_length = config.get('min_length', 15)
        self.max_length = config.get('max_length', 350)
        self.min_devanagari_ratio = config.get('min_devanagari_ratio', 0.30)
        self.norm_form = config.get('norm_form', 'NFC')
        
        logger.info(
            f"NepaliTextProcessor initialized: "
            f"min_len={self.min_length}, max_len={self.max_length}, "
            f"min_dev_ratio={self.min_devanagari_ratio}"
        )
    
    def clean_and_normalize(self, text: str) -> str:
        """
        Apply Unicode NFC normalization, clean invisible characters, and normalize whitespace.
        
        Args:
            text: Raw input string
            
        Returns:
            Cleaned and NFC-normalized string
        """
        return normalize_nepali_unicode(text)
    
    def segment_sentences(self, paragraph: str) -> List[str]:
        """
        Segment a multi-sentence paragraph into individual sentences without dropping trailing content.
        
        Supports:
        - Purna Viram (।)
        - Double Purna Viram (॥)
        - Question marks (?)
        - Exclamation marks (!)
        - Newline boundaries (\n)
        
        Boundary guarantees:
        - Preserves sentences even if trailing delimiter is absent.
        - Delimiter clusters (e.g. ??, !!, ।।) remain attached to the preceding sentence.
        - Enforces inclusive boundary constraints: min_length <= len(s) <= max_length.
        
        Args:
            paragraph: Raw multi-sentence text
            
        Returns:
            List of segmented, validated sentences
        """
        if not paragraph:
            return []
        
        # Pre-normalize Unicode NFC and clean zero-width characters
        text = unicodedata.normalize(self.norm_form, paragraph)
        text = clean_zero_width_characters(text)
        
        # Split on newlines first to respect explicit paragraph/line boundaries
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        raw_sentences: List[str] = []
        
        for line in lines:
            # Split keeping delimiters: alternating [content, delim, content, delim, ...]
            parts = SENTENCE_TERMINATOR_PATTERN.split(line)
            
            current_sentence = ""
            for part in parts:
                if not part:
                    continue
                # If part is a sentence terminator cluster, attach it to current sentence
                if SENTENCE_TERMINATOR_PATTERN.fullmatch(part):
                    current_sentence = (current_sentence + part).strip()
                    if current_sentence:
                        raw_sentences.append(current_sentence)
                        current_sentence = ""
                else:
                    # Content chunk
                    cleaned_part = WHITESPACE_PATTERN.sub(' ', part).strip()
                    if cleaned_part:
                        if current_sentence:
                            current_sentence = current_sentence + " " + cleaned_part
                        else:
                            current_sentence = cleaned_part
            
            # Handle trailing text that lacked terminal punctuation
            if current_sentence.strip():
                raw_sentences.append(current_sentence.strip())
        
        # Filter sentences by length constraints and Devanagari ratio
        validated_sentences = []
        for s in raw_sentences:
            s_clean = WHITESPACE_PATTERN.sub(' ', s).strip()
            if self.min_length <= len(s_clean) <= self.max_length:
                if self.is_valid_sentence(s_clean):
                    validated_sentences.append(s_clean)
        
        return validated_sentences
    
    def is_valid_sentence(self, sentence: str) -> bool:
        """
        Validate whether a sentence meets length and Devanagari script criteria.
        
        Args:
            sentence: Sentence string to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not sentence:
            return False
        if not (self.min_length <= len(sentence) <= self.max_length):
            return False
        return devanagari_ratio(sentence) >= self.min_devanagari_ratio
    
    def compute_hash(self, sentence: str) -> str:
        """
        Compute SHA-256 hash for sentence deduplication.
        
        Args:
            sentence: Sentence string
            
        Returns:
            Hexadecimal SHA-256 string
        """
        return compute_sentence_hash(sentence)
    
    def batch_process(self, texts: List[str], mode: str = "segment") -> List[str]:
        """
        Apply processing to multiple texts in batch.
        
        Args:
            texts: List of raw text strings
            mode: "segment" (return all sentences) or "normalize" (return cleaned texts)
            
        Returns:
            List of processed strings
        """
        results: List[str] = []
        for text in texts:
            if mode == "segment":
                results.extend(self.segment_sentences(text))
            elif mode == "normalize":
                cleaned = self.clean_and_normalize(text)
                if cleaned:
                    results.append(cleaned)
            else:
                raise ValueError(f"Unknown mode: {mode}. Must be 'segment' or 'normalize'")
        return results
    
    def get_statistics(self, texts: List[str]) -> Dict:
        """
        Compute processing statistics across a batch of texts.
        
        Args:
            texts: List of raw input texts
            
        Returns:
            Dictionary with counts, lengths, and compression ratios
        """
        segmented: List[str] = []
        for text in texts:
            segmented.extend(self.segment_sentences(text))
        
        if not segmented:
            return {
                "input_texts": len(texts),
                "output_sentences": 0,
                "avg_length": 0.0,
                "min_length": 0,
                "max_length": 0,
                "compression_ratio": 0.0
            }
        
        lengths = [len(s) for s in segmented]
        return {
            "input_texts": len(texts),
            "output_sentences": len(segmented),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "compression_ratio": len(segmented) / max(1, len(texts))
        }
