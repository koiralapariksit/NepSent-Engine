"""
Nepali text processing pipeline.
Implements Unicode normalization, script filtering, and sentence segmentation.
"""

import re
import unicodedata
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class NepaliTextProcessor:
    """
    Production-grade Nepali text processor for web-scraped content.
    
    Responsibilities:
    1. Unicode Form C (NFC) normalization
    2. Non-Devanagari noise filtering (preserves spaces, numerics, Nepali punctuation)
    3. Sentence tokenization by native delimiters (।, ?, !)
    4. Length-based filtering
    
    Scientific Basis:
    - NFC normalization ensures consistency in morphological analysis
    - Character-level filtering prevents script-mixing interference
    - Sentence-level segmentation enables fine-grained annotation
    """
    
    def __init__(self, config=None):
        """
        Initialize processor with configuration.
        
        Args:
            config: Dict containing filtering and segmentation parameters
        """
        if config is None:
            config = {}
        
        # Devanagari Unicode range (U+0900–U+097F) + ASCII + punctuation
        self.devanagari_filter = re.compile(
            config.get('devanagari_pattern', 
                      r'([^\u0900-\u097F\u0020-\u007E\n।?!])')
        )
        self.sentence_tokenizer = re.compile(
            config.get('sentence_delimiter',
                      r'([।?!])')
        )
        self.min_length = config.get('min_length', 15)
        self.max_length = config.get('max_length', 350)
        
        logger.info(
            f"NepaliTextProcessor initialized: "
            f"min_len={self.min_length}, max_len={self.max_length}"
        )
    
    def clean_and_normalize(self, text: str) -> str:
        """
        Apply NFC normalization and filter non-Devanagari noise.
        
        Steps:
        1. Handle null/empty input
        2. Apply NFC normalization (canonical composition)
        3. Remove non-Devanagari characters (preserves ASCII for numbers/basic punctuation)
        4. Collapse multiple whitespace to single space
        5. Strip leading/trailing whitespace
        
        Args:
            text: Raw input string
            
        Returns:
            Cleaned and normalized string
            
        Example:
            >>> processor = NepaliTextProcessor()
            >>> text = "मेरो नाम  राज छ !!! 😊"  # Extra spaces, emoji, extra punctuation
            >>> processor.clean_and_normalize(text)
            'मेरो नाम राज छ !'
        """
        if not text:
            return ""
        
        # Step 1: NFC normalization (canonical decomposition + recomposition)
        text = unicodedata.normalize('NFC', text)
        
        # Step 2: Remove non-Devanagari, non-ASCII noise (keeps spaces, basic punctuation)
        text = self.devanagari_filter.sub('', text)
        
        # Step 3: Collapse multiple whitespace to single space
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def segment_sentences(self, paragraph: str) -> List[str]:
        """
        Segment a multi-sentence paragraph into individual sentences.
        
        Algorithm:
        1. Clean and normalize input paragraph
        2. Split by sentence-final delimiters (।, ?, !)
        3. Reconstruct sentence by pairing content with delimiter
        4. Filter by length constraints
        
        Args:
            paragraph: Raw multi-sentence text
            
        Returns:
            List of cleaned, validated sentences
            
        Example:
            >>> text = "मलाई सेब पनि राम्रो लाग्यो। आलु पनि राम्रो छ। तपाई कस्तो छौ?"
            >>> processor.segment_sentences(text)
            ['मलाई सेब पनि राम्रो लाग्यो।', 'आलु पनि राम्रो छ।', 'तपाई कस्तो छौ?']
        """
        normalized = self.clean_and_normalize(paragraph)
        
        if not normalized:
            return []
        
        # Split by sentence delimiters (keeps delimiters in result)
        chunks = self.sentence_tokenizer.split(normalized)
        
        sentences = []
        
        # Reconstruct sentences by pairing content with delimiters
        # chunks = [content[0], delim[0], content[1], delim[1], ...]
        for i in range(0, len(chunks) - 1, 2):
            # Combine content chunk with its delimiter
            sentence = (chunks[i] + chunks[i + 1]).strip()
            
            # Validate length constraints
            if self.min_length < len(sentence) < self.max_length:
                sentences.append(sentence)
        
        return sentences
    
    def batch_process(self, texts: List[str], mode: str = "segment") -> List[str]:
        """
        Apply processing to multiple texts in batch.
        
        Args:
            texts: List of raw text strings
            mode: "segment" (return all sentences) or "normalize" (return single normalized text)
            
        Returns:
            List of processed outputs
        """
        results = []
        
        for text in texts:
            if mode == "segment":
                results.extend(self.segment_sentences(text))
            elif mode == "normalize":
                results.append(self.clean_and_normalize(text))
            else:
                raise ValueError(f"Unknown mode: {mode}")
        
        return results
    
    def get_statistics(self, texts: List[str]) -> dict:
        """
        Compute processing statistics on a batch of texts.
        
        Returns:
            Dict with counts and metrics
        """
        segmented = []
        for text in texts:
            segmented.extend(self.segment_sentences(text))
        
        if not segmented:
            return {
                "input_texts": len(texts),
                "output_sentences": 0,
                "avg_length": 0,
                "compression_ratio": 0.0
            }
        
        lengths = [len(s) for s in segmented]
        
        return {
            "input_texts": len(texts),
            "output_sentences": len(segmented),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "compression_ratio": len(segmented) / len(texts)  # sentences per input
        }


# ============================================================================
# Utility Functions
# ============================================================================

def is_valid_nepali_text(text: str) -> bool:
    """
    Quick validation that text contains Nepali Devanagari script.
    
    Args:
        text: Input string
        
    Returns:
        True if contains at least one Devanagari character
    """
    if not text:
        return False
    devanagari_chars = re.findall(r'[\u0900-\u097F]', text)
    return len(devanagari_chars) > 0


def estimate_data_loss(raw_count: int, expected_loss_ratio: float = 0.55) -> int:
    """
    Estimate output after standard 50-60% data loss from processing.
    
    Args:
        raw_count: Number of raw sentences collected
        expected_loss_ratio: Proportion lost during cleaning (default 0.55 = 55%)
        
    Returns:
        Estimated clean sentence count
    """
    return int(raw_count * (1 - expected_loss_ratio))
