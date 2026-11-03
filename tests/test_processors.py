"""
Unit tests for nepali_scraper/processors.py
Covers Unicode NFC normalization, zero-width character handling,
Devanagari sentence segmentation edge cases, inclusive length filtering,
Devanagari ratio validation, and SHA-256 deduplication hashing.
"""

import sys
import unicodedata
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nepali_scraper.processors import (
    NepaliTextProcessor,
    clean_zero_width_characters,
    compute_sentence_hash,
    devanagari_ratio,
    is_valid_nepali_text,
)


# ============================================================================
# 1. UNICODE NFC NORMALIZATION TESTS
# ============================================================================

def test_unicode_nfc_normalization():
    """Verify that decomposed text is canonically composed into NFC form."""
    processor = NepaliTextProcessor()
    # Test decomposed Latin diacritic in Romanized Nepali (a + combining macron -> ā)
    decomposed_iast = "Nep\u0061\u0304l"  # NFD for "Nepāl"
    composed_iast = "Nepāl"
    assert decomposed_iast != composed_iast
    assert len(decomposed_iast) == 6
    assert len(composed_iast) == 5
    
    normalized_iast = processor.clean_and_normalize(decomposed_iast)
    assert normalized_iast == composed_iast
    assert unicodedata.is_normalized('NFC', normalized_iast)

    # Test Devanagari text normalization
    nepali_text = "नेपालको नयाँ संविधान"
    normalized_dev = processor.clean_and_normalize(nepali_text)
    assert normalized_dev == nepali_text
    assert unicodedata.is_normalized('NFC', normalized_dev)


def test_nfc_normalization_preserves_virama():
    """Verify that virama (्, U+094D) and halant conjuncts are preserved under NFC."""
    processor = NepaliTextProcessor()
    text = "नमस्कार, म काठमाडौँबाट हुँ।"
    normalized = processor.clean_and_normalize(text)
    assert "्" in normalized  # Virama preserved
    assert normalized == text


# ============================================================================
# 2. ZERO-WIDTH CHARACTER HANDLING TESTS
# ============================================================================

def test_clean_non_semantic_zero_width_spaces():
    """Verify that ZWSP, BOM, and direction marks are cleanly removed."""
    # Insert \u200B (ZWSP), \uFEFF (BOM), \u200E (LRM), \u200F (RLM), \u00AD (soft hyphen)
    dirty = "ने\u200Bपा\uFEFFल\u200Eको\u200F स\u00ADरकार"
    cleaned = clean_zero_width_characters(dirty)
    assert cleaned == "नेपालको सरकार"


def test_preserve_devanagari_zwj_and_zwnj():
    """Verify that ZWJ (\\u200D) and ZWNJ (\\u200C) are preserved for conjuncts."""
    # Eyelash-ra in Nepali: र् + ZWJ (र + ् + ZWJ)
    eyelash_ra = "र्यापिड र्‍याली"
    cleaned = clean_zero_width_characters(eyelash_ra)
    assert "\u200D" in cleaned
    assert cleaned == eyelash_ra


def test_collapse_consecutive_zero_width_joiners():
    """Verify duplicate ZWJ or ZWNJ sequences are collapsed to a single character."""
    consecutive_zwj = "नेपाल\u200D\u200D\u200D"
    cleaned = clean_zero_width_characters(consecutive_zwj)
    assert cleaned == "नेपाल\u200D"


# ============================================================================
# 3. SENTENCE SEGMENTATION & DELIMITER EDGE CASES
# ============================================================================

def test_sentence_segmentation_with_purna_viram():
    """Verify segmentation on standard Purna Viram (।)."""
    processor = NepaliTextProcessor(config={'min_length': 10})
    text = "नेपाल एक सुन्दर देश हो। पोखरा प्राकृतिक रूपमा धनी छ।"
    sentences = processor.segment_sentences(text)
    assert len(sentences) == 2
    assert sentences[0] == "नेपाल एक सुन्दर देश हो।"
    assert sentences[1] == "पोखरा प्राकृतिक रूपमा धनी छ।"


def test_sentence_segmentation_with_question_and_exclamation():
    """Verify segmentation on question mark (?) and exclamation mark (!)."""
    processor = NepaliTextProcessor(config={'min_length': 5})
    text = "तपाईंलाई कस्तो छ? म एकदम ठीक छु! धन्यवाद।"
    sentences = processor.segment_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "तपाईंलाई कस्तो छ?"
    assert sentences[1] == "म एकदम ठीक छु!"
    assert sentences[2] == "धन्यवाद।"


def test_sentence_segmentation_with_double_danda():
    """Verify segmentation on deergha viram / double danda (॥)."""
    processor = NepaliTextProcessor(config={'min_length': 5})
    text = "प्रथम श्लोक समाप्त भयो॥ दोस्रो श्लोक यहाँ छ॥"
    sentences = processor.segment_sentences(text)
    assert len(sentences) == 2
    assert sentences[0] == "प्रथम श्लोक समाप्त भयो॥"
    assert sentences[1] == "दोस्रो श्लोक यहाँ छ॥"


def test_sentence_segmentation_preserves_trailing_sentence_without_delimiter():
    """
    CRITICAL EDGE CASE: Paragraph lacking a terminal delimiter.
    Must NOT be silently dropped.
    """
    processor = NepaliTextProcessor(config={'min_length': 10})
    text = "नेपालको अर्थतन्त्र सुधारको बाटोमा छ"
    sentences = processor.segment_sentences(text)
    assert len(sentences) == 1
    assert sentences[0] == "नेपालको अर्थतन्त्र सुधारको बाटोमा छ"


def test_sentence_segmentation_preserves_trailing_sentence_after_delimited_one():
    """
    CRITICAL EDGE CASE: A delimited sentence followed by a sentence without delimiter.
    Both must be preserved.
    """
    processor = NepaliTextProcessor(config={'min_length': 10})
    text = "पहिलो वाक्य समाप्त भयो। तर दोस्रो वाक्यमा पूर्णविराम छैन"
    sentences = processor.segment_sentences(text)
    assert len(sentences) == 2
    assert sentences[0] == "पहिलो वाक्य समाप्त भयो।"
    assert sentences[1] == "तर दोस्रो वाक्यमा पूर्णविराम छैन"


def test_sentence_segmentation_with_repeated_delimiters():
    """Verify handling of delimiter clusters like ?? or !! or ।।।."""
    processor = NepaliTextProcessor(config={'min_length': 5})
    text = "के यो साँच्चै सत्य हो?? अवश्य पनि सत्य हो!!"
    sentences = processor.segment_sentences(text)
    assert len(sentences) == 2
    assert sentences[0] == "के यो साँच्चै सत्य हो??"
    assert sentences[1] == "अवश्य पनि सत्य हो!!"


def test_sentence_segmentation_with_newlines():
    """Verify that newline breaks act as valid sentence boundaries."""
    processor = NepaliTextProcessor(config={'min_length': 10})
    text = "काठमाडौंमा आज पानी पर्यो\nभोलि मौसम सफा हुनेछ"
    sentences = processor.segment_sentences(text)
    assert len(sentences) == 2
    assert sentences[0] == "काठमाडौंमा आज पानी पर्यो"
    assert sentences[1] == "भोलि मौसम सफा हुनेछ"


# ============================================================================
# 4. INCLUSIVE BOUNDARY LENGTH FILTERING TESTS
# ============================================================================

def test_inclusive_min_length_boundary():
    """Verify that a sentence of length exactly equal to min_length is included."""
    # Length exactly 15 characters
    sentence_15_chars = "नेपाल सुन्दर छ।"  # len("नेपाल सुन्दर छ।") == 15
    assert len(sentence_15_chars) == 15
    
    processor = NepaliTextProcessor(config={'min_length': 15, 'max_length': 50})
    sentences = processor.segment_sentences(sentence_15_chars)
    assert len(sentences) == 1
    assert sentences[0] == sentence_15_chars


def test_below_min_length_is_excluded():
    """Verify that a sentence of length < min_length is excluded."""
    short_sentence = "राम्रो छ।"  # len == 9
    processor = NepaliTextProcessor(config={'min_length': 15, 'max_length': 50})
    sentences = processor.segment_sentences(short_sentence)
    assert len(sentences) == 0


def test_inclusive_max_length_boundary():
    """Verify that a sentence of length exactly equal to max_length is included."""
    # Create valid Devanagari sentence of exactly 30 characters
    base = "नेपाल देश धेरै सुन्दर छ आज।"  # len == 28
    padded = base + "।। "  # len == 30 when trimmed without trailing space
    padded_clean = padded.strip()
    target_len = len(padded_clean)
    
    processor = NepaliTextProcessor(config={'min_length': 10, 'max_length': target_len})
    sentences = processor.segment_sentences(padded_clean)
    assert len(sentences) == 1


# ============================================================================
# 5. DEVANAGARI CHARACTER RATIO & SCRIPT VALIDATION
# ============================================================================

def test_devanagari_ratio_pure_nepali():
    """Verify ratio is 1.0 for pure Devanagari text."""
    text = "नेपालको प्राकृतिक सौन्दर्य"
    assert devanagari_ratio(text) == 1.0
    assert is_valid_nepali_text(text)


def test_devanagari_ratio_code_mixed():
    """Verify code-mixed text with English brand name passes the 0.30 threshold."""
    text = "नयाँ iPhone 15 Pro बजारमा उपलब्ध छ।"
    ratio = devanagari_ratio(text)
    assert 0.30 <= ratio <= 0.90
    assert is_valid_nepali_text(text, min_ratio=0.30)


def test_devanagari_ratio_pure_english():
    """Verify pure English text has ratio 0.0 and fails validation."""
    text = "Breaking news headline about economy"
    assert devanagari_ratio(text) == 0.0
    assert not is_valid_nepali_text(text, min_ratio=0.30)


def test_empty_string_devanagari_ratio():
    """Verify empty string returns 0.0 ratio."""
    assert devanagari_ratio("") == 0.0
    assert devanagari_ratio("   \t\n  ") == 0.0


# ============================================================================
# 6. SHA-256 SENTENCE HASHING & DEDUPLICATION TESTS
# ============================================================================

def test_hash_invariance_to_whitespace():
    """Verify that sentences differing only by whitespace yield identical SHA-256 hashes."""
    s1 = "नेपालको राजधानी काठमाडौँ हो।"
    s2 = "   नेपालको   राजधानी    काठमाडौँ   हो।   "
    s3 = "नेपालको\tराजधानी\nकाठमाडौँ हो।"
    
    h1 = compute_sentence_hash(s1)
    h2 = compute_sentence_hash(s2)
    h3 = compute_sentence_hash(s3)
    
    assert h1 == h2
    assert h2 == h3
    assert len(h1) == 64  # Valid SHA-256 hex string


def test_hash_invariance_to_unicode_decomposition():
    """Verify that NFD and NFC forms of the same sentence yield identical SHA-256 hashes."""
    s_nfc = "सरकारले नयाँ शिक्षा नीति घोषणा गर्‍यो।"
    s_nfd = unicodedata.normalize('NFD', s_nfc)
    
    assert compute_sentence_hash(s_nfc) == compute_sentence_hash(s_nfd)


def test_distinct_sentences_yield_distinct_hashes():
    """Verify that semantically distinct sentences yield different hashes."""
    s1 = "यो किताब धेरै राम्रो छ।"
    s2 = "यो किताब धेरै नराम्रो छ।"
    assert compute_sentence_hash(s1) != compute_sentence_hash(s2)


def test_processor_instance_hash_matches_standalone():
    """Verify processor.compute_hash matches compute_sentence_hash function."""
    processor = NepaliTextProcessor()
    text = "मलाई नेपाली भाषा मन पर्छ।"
    assert processor.compute_hash(text) == compute_sentence_hash(text)


# ============================================================================
# 7. BATCH PROCESSING & PROCESSING STATISTICS TESTS
# ============================================================================

def test_batch_process_segmentation():
    """Verify batch processing produces concatenated segmented sentences."""
    processor = NepaliTextProcessor(config={'min_length': 10})
    texts = [
        "पहिलो अनुच्छेदको वाक्य हो यो।",
        "दोस्रो अनुच्छेदको पहिलो वाक्य। दोस्रो अनुच्छेदको दोस्रो वाक्य।"
    ]
    results = processor.batch_process(texts, mode="segment")
    assert len(results) == 3


def test_batch_process_normalization():
    """Verify batch processing in normalize mode returns cleaned paragraphs."""
    processor = NepaliTextProcessor()
    texts = [
        "  पहिलो   वाक्य  ।  ",
        "दोस्रो\u200B वाक्य ।"
    ]
    results = processor.batch_process(texts, mode="normalize")
    assert len(results) == 2
    assert results[0] == "पहिलो वाक्य ।"
    assert results[1] == "दोस्रो वाक्य ।"


def test_processor_statistics():
    """Verify processing statistics computation."""
    processor = NepaliTextProcessor(config={'min_length': 10})
    texts = [
        "पहिलो अनुच्छेदको वाक्य हो यो।",
        "दोस्रो अनुच्छेदको पहिलो वाक्य। दोस्रो अनुच्छेदको दोस्रो वाक्य।"
    ]
    stats = processor.get_statistics(texts)
    assert stats["input_texts"] == 2
    assert stats["output_sentences"] == 3
    assert stats["min_length"] > 0
    assert stats["max_length"] >= stats["min_length"]
    assert stats["compression_ratio"] == 1.5
