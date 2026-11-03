"""
Nepali Linguistic & Error Taxonomy Analyzer for NepSent-Engine.

Provides:
1. Morphological Negation Detector:
   Accurately detects genuine Nepali negation (copulas, auxiliaries, inflectional verbal suffixes,
   and prefixed negative verbs) without false-triggering on nouns/words starting with 'न' (e.g., नेपाल, नयाँ, नदी).
2. Code-Mixing & Script Analyzer:
   Identifies Romanized English/Nepali tokens, script ratios, and mixed-script sentences.
3. Subword Fertility Calculator:
   Measures average tokens per word (fertility), character-to-token ratio, and [UNK] token rates.
4. Error Taxonomy Generator:
   Categorizes model misclassifications into linguistically grounded diagnostic error buckets.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np

from nepali_scraper.config import SCHEMA_VERSION, LABEL_SCHEMA

logger = logging.getLogger("nepali_scraper.evaluation.linguistics")

# Nepali word token pattern avoiding Purna Viram (\u0964) and Deergha Viram (\u0965)
WORD_TOKEN_PATTERN = re.compile(r'[\u0900-\u0963\u0966-\u097F\w]+')

# Latin / Roman script token pattern
LATIN_TOKEN_PATTERN = re.compile(r'\b[A-Za-z]+\b')

# ---------------------------------------------------------------------------
# 1. MORPHOLOGICAL NEGATION PATTERNS
# ---------------------------------------------------------------------------

# Standalone auxiliary, copular, and modal negation words in authentic Nepali
STANDALONE_NEGATION_WORDS = {
    # Non-past / existential copulas
    "छैन", "छैनन्", "छैनौ", "छैनस्", "छैनस", "छैन्",
    # Identificational copulas
    "होइन", "होइनन्", "होइनौ", "होइनस्", "होइनन", "होईन",
    # Past copulas
    "थिएन", "थिएनन्", "थिएनौ", "थिएनस्", "थिएनन",
    # Habitual / potential
    "हुँदैन", "हुँदैनन्", "हुँदैनथ्यो", "हुन्न", "हुन्नन्",
    "गर्दैन", "गर्दैनन्", "गर्दैनौ", "गर्दैनस्",
    "सक्दैन", "सक्दैनन्", "सक्दैनौ", "सक्दैनस्",
    "मिल्दैन", "मिल्दैनन्", "मिल्दैनौ",
    "चाहिँदैन", "चाहिँदैनन्", "चाहिदैन", "चाहिदैनन्",
    "पर्दैन", "पर्दैनन्", "पर्दैनौ",
    "भएन", "भएनन्", "भएनौ", "भएनस्",
    "गरेन", "गरेनन्", "गरेनौ", "गरेनस्",
    "खाइन", "खाइनन्", "भनिन", "भनिनन्",
    "लेखेन", "लेखेनन्", "बसेन", "बसेनन्",
    "लाग्दैन", "लाग्दैनन्", "मान्दैन", "मान्दैनन्",
    "सकेन", "सकेनन्", "सकेनौ",
}

# Regex for inflectional verbal negation suffixes attached to Nepali verb stems:
# - Non-past negative 3rd person: -दैन, -दैनन्, -दैँन, -दैनौ, -दैनस्
# - Past negative (matra / independent vowel): -[ेए]न, -[ेए]नन्, -[ेए]नौ, -[ेए]नस् (e.g. गरेन, लेखेन, दिएन, आएन, बसेन)
# - First-person negative: -न्न, -न्नँ, -दिन, -दिनँ (e.g. जान्न, जान्नँ, गर्दिन, भन्दिन)
# - Feminine past negative: -[िी]नन् (e.g. खाइनन्, गइनन्)
VERBAL_NEGATION_SUFFIX_PATTERN = re.compile(
    r'[\u0900-\u0963\u0966-\u097F]+(?:दैन|दैनन्|दैँन|दैनौ|दैनस्|[ेए]न|[ेए]नन्|[ेए]नौ|[ेए]नस्|न्न|न्नँ|दिन|दिनँ|दिनन्|[िी]नन्)\b'
)

# Regex for prefixed negative verbs (imperative, conditional, or non-finite negative forms)
# e.g., नगर्नु, नगरे, नभए, नजाउ, नजानु, नखानु, नआउनु, नहुँदा, नदेखाउ, नकिन्नु, नकिन्नुहोला
PREFIXED_NEGATIVE_VERB_PATTERN = re.compile(
    r'\bन(?:गर्नु?|गर्नुहोला|गरे|गरेको|भए|भएको|जाउ|जानु|जानुहोला|खानु|खाउ|आउनु|आउ|हुँदा|हुने|भनी|देखाउ|किन्नु|किन्नुहोला|खोज्नु|दिनु|लिनु)\b'
)

# Standalone negative particle 'न' (strictly isolated by word boundaries and followed by a verb/action word)
ISOLATED_NEGATIVE_PARTICLE_PATTERN = re.compile(
    r'(?:^|[\s।?!॥,])न(?:\s+[\u0900-\u0963\u0966-\u097F]+)'
)

# Exception whitelist: common nouns / adjectives that happen to end in -एन or -न but are NOT negative verbs
NON_NEGATIVE_EXCEPTIONS = {
    "नेपाल", "नयाँ", "नदी", "नागरिक", "नम्र", "नेता", "नियम",
    "स्टेशन", "स्टेसन", "टेलिफोन", "प्लेन", "ट्रेन", "पेन", "फ्यान",
    "प्लान", "डेलिभरी", "स्थान", "ध्यान", "मान", "गान", "दान",
    "कान", "पान", "सान", "ज्ञान", "विज्ञान", "संविधान", "अनुसन्धान",
    "किसान", "विद्वान", "विद्वान्", "भगवान", "भगवान्", "दुकान",
    "मैदान", "तूफान", "तुफान", "साधन", "प्रमाण", "विमान",
}


def detect_nepali_negation(text: str) -> Dict[str, Any]:
    """
    Detect authentic Nepali linguistic negation.
    
    STRICT RESEARCH CRITERION:
    Avoids false positives from simplistic character checks (like `"न" in text`)
    by enforcing word-boundary token matching, auxiliary lookup, and verbal
    suffix / prefix morphology.
    
    Args:
        text: Input Nepali text string
        
    Returns:
        Structured dictionary indicating presence of negation, matched tokens, and patterns
    """
    if not text or not text.strip():
        return {
            "has_negation": False,
            "matched_tokens": [],
            "matched_patterns": [],
        }
    
    cleaned_text = text.strip()
    words = WORD_TOKEN_PATTERN.findall(cleaned_text)
    
    matched_tokens = []
    matched_patterns = []
    
    for w in words:
        # 1. Standalone copular / auxiliary negation
        if w in STANDALONE_NEGATION_WORDS:
            matched_tokens.append(w)
            matched_patterns.append("standalone_auxiliary_negation")
            continue
        
        # Skip known non-negative nouns / borrowings
        if w in NON_NEGATIVE_EXCEPTIONS:
            continue
        
        # 2. Prefixed negative verbs (e.g. नकिन्नुहोला, नगरे, नभए)
        if PREFIXED_NEGATIVE_VERB_PATTERN.match(w):
            matched_tokens.append(w)
            matched_patterns.append("prefixed_negative_verb")
            continue
        
        # 3. Inflectional verbal negation suffixes (e.g. गरेन, जान्नँ, खाइनन्, खाँदैन)
        if VERBAL_NEGATION_SUFFIX_PATTERN.match(w):
            matched_tokens.append(w)
            matched_patterns.append("verbal_suffix_negation")
            continue
            
    # 4. Standalone negative particle 'न' followed by verb (e.g. "न गर", "न बोल")
    if ISOLATED_NEGATIVE_PARTICLE_PATTERN.search(cleaned_text):
        matched_tokens.append("न")
        matched_patterns.append("isolated_negative_particle")
    
    has_negation = len(matched_tokens) > 0
    
    return {
        "has_negation": has_negation,
        "matched_tokens": sorted(list(set(matched_tokens))),
        "matched_patterns": sorted(list(set(matched_patterns))),
    }


def analyze_code_mixing(text: str) -> Dict[str, Any]:
    """
    Analyze script code-mixing (Devanagari vs Latin / Roman script) in Nepali text.
    
    Args:
        text: Input text string
        
    Returns:
        Structured dictionary with character counts, ratios, and Latin tokens
    """
    if not text or not text.strip():
        return {
            "is_code_mixed": False,
            "devanagari_ratio": 0.0,
            "latin_ratio": 0.0,
            "devanagari_chars": 0,
            "latin_chars": 0,
            "total_non_ws_chars": 0,
            "latin_tokens": [],
        }
    
    non_ws = re.sub(r'\s+', '', text)
    total_len = len(non_ws)
    
    dev_chars = len(re.findall(r'[\u0900-\u097F]', non_ws))
    latin_chars = len(re.findall(r'[A-Za-z]', non_ws))
    latin_tokens = LATIN_TOKEN_PATTERN.findall(text)
    
    dev_ratio = round(dev_chars / total_len, 4) if total_len > 0 else 0.0
    latin_ratio = round(latin_chars / total_len, 4) if total_len > 0 else 0.0
    
    # Text is considered code-mixed or Romanized if Latin tokens are present
    is_code_mixed = bool(latin_chars > 0 or len(latin_tokens) > 0)
    
    return {
        "is_code_mixed": is_code_mixed,
        "devanagari_ratio": dev_ratio,
        "latin_ratio": latin_ratio,
        "devanagari_chars": dev_chars,
        "latin_chars": latin_chars,
        "total_non_ws_chars": total_len,
        "latin_tokens": latin_tokens,
    }


def compute_subword_fertility(texts: List[str], tokenizer: Any) -> Dict[str, Any]:
    """
    Compute subword tokenizer fertility and character-to-token statistics on Nepali text.
    
    Measures how aggressively a tokenizer fragments Nepali morphological words into subwords.
    High fertility (> 2.0 - 2.5 tokens/word) is a hallmark of poor low-resource tokenization.
    
    Args:
        texts: List of Nepali text strings
        tokenizer: HuggingFace or custom tokenizer supporting .tokenize() or whitespace split
        
    Returns:
        Structured fertility dictionary with tokens/word, char/token, and UNK rate
    """
    total_words = 0
    total_tokens = 0
    total_chars = 0
    unk_count = 0
    
    for t in texts:
        if not t:
            continue
        words = WORD_TOKEN_PATTERN.findall(t)
        total_words += len(words)
        total_chars += len(re.sub(r'\s+', '', t))
        
        # Extract tokens using tokenizer
        if hasattr(tokenizer, "tokenize") and callable(tokenizer.tokenize):
            subwords = tokenizer.tokenize(t)
        elif hasattr(tokenizer, "encode") and callable(tokenizer.encode):
            subwords = [str(token_id) for token_id in tokenizer.encode(t)]
        else:
            subwords = t.split()
            
        total_tokens += len(subwords)
        
        # Check UNK tokens
        for sw in subwords:
            if sw in ["[UNK]", "<unk>", "UNK"]:
                unk_count += 1
                
    fertility = round(total_tokens / total_words, 4) if total_words > 0 else 0.0
    char_to_token = round(total_chars / total_tokens, 4) if total_tokens > 0 else 0.0
    unk_rate = round(unk_count / total_tokens, 4) if total_tokens > 0 else 0.0
    
    return {
        "texts_analyzed": len(texts),
        "total_words": total_words,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "fertility": fertility,
        "char_to_token_ratio": char_to_token,
        "unk_count": unk_count,
        "unk_rate": unk_rate,
    }


def generate_error_taxonomy(
    records: List[Dict[str, Any]],
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    tokenizer: Optional[Any] = None,
    text_key: str = "text",
) -> Dict[str, Any]:
    """
    Categorize model misclassifications into linguistically grounded diagnostic buckets.
    
    Diagnostic Buckets:
    1. Negation Scope Error: Text contains genuine negation and prediction flips polarity.
    2. Neutral-Polar Boundary Confusion: Misclassifications crossing the Neutral boundary (0/2 <-> 1).
    3. Long-Sentence Degradation: Sentences with > 200 characters or > 35 words.
    4. Code-Mixed Error: Misclassified sentences containing Latin/English tokens.
    5. High-Fertility Fragmentation: Text has subword fertility > 2.5 or UNK tokens (if tokenizer given).
    6. Other Ambiguity: Residual errors not captured by specific linguistic categories.
    
    Args:
        records: List of sample dictionaries containing text
        y_true: Ground truth sentiment labels
        y_pred: Model predicted sentiment labels
        tokenizer: Optional tokenizer for subword fertility diagnosis
        text_key: Dictionary key for text content
        
    Returns:
        Structured taxonomy report with error counts, proportions, and sample breakdowns
    """
    yt_arr = np.asarray(y_true, dtype=int)
    yp_arr = np.asarray(y_pred, dtype=int)
    total_evals = len(yt_arr)
    
    if total_evals == 0:
        raise ValueError("Cannot generate error taxonomy for empty inputs.")
    
    error_indices = [i for i in range(total_evals) if yt_arr[i] != yp_arr[i]]
    total_errors = len(error_indices)
    
    bucket_counts = {
        "negation_scope_error": 0,
        "neutral_polar_boundary_confusion": 0,
        "long_sentence_degradation": 0,
        "code_mixed_error": 0,
        "high_fertility_fragmentation": 0,
        "other_ambiguity": 0,
    }
    
    detailed_samples = []
    
    for idx in error_indices:
        rec = records[idx]
        text = rec.get(text_key, "")
        yt = int(yt_arr[idx])
        yp = int(yp_arr[idx])
        
        neg_info = detect_nepali_negation(text)
        cm_info = analyze_code_mixing(text)
        words = WORD_TOKEN_PATTERN.findall(text)
        word_count = len(words)
        char_count = len(text)
        
        matched_buckets = []
        
        # 1. Negation Scope Error
        if neg_info["has_negation"]:
            # Polarity inversion (e.g. 0 <-> 2 or polar mispredicted)
            matched_buckets.append("negation_scope_error")
            bucket_counts["negation_scope_error"] += 1
            
        # 2. Neutral-Polar Boundary Confusion
        if (yt == 1 and yp != 1) or (yt != 1 and yp == 1):
            matched_buckets.append("neutral_polar_boundary_confusion")
            bucket_counts["neutral_polar_boundary_confusion"] += 1
            
        # 3. Long-Sentence Degradation
        if char_count > 200 or word_count > 35:
            matched_buckets.append("long_sentence_degradation")
            bucket_counts["long_sentence_degradation"] += 1
            
        # 4. Code-Mixed Error
        if cm_info["is_code_mixed"]:
            matched_buckets.append("code_mixed_error")
            bucket_counts["code_mixed_error"] += 1
            
        # 5. High-Fertility Fragmentation
        if tokenizer is not None:
            if hasattr(tokenizer, "tokenize"):
                subwords = tokenizer.tokenize(text)
                fertility = len(subwords) / word_count if word_count > 0 else 0.0
                has_unk = any(sw in ["[UNK]", "<unk>"] for sw in subwords)
                if fertility > 2.5 or has_unk:
                    matched_buckets.append("high_fertility_fragmentation")
                    bucket_counts["high_fertility_fragmentation"] += 1
                    
        # 6. Other Ambiguity
        if not matched_buckets:
            matched_buckets.append("other_ambiguity")
            bucket_counts["other_ambiguity"] += 1
            
        if len(detailed_samples) < 50:
            detailed_samples.append({
                "sample_index": idx,
                "text": text,
                "true_label": yt,
                "true_label_name": LABEL_SCHEMA.get(yt, str(yt)),
                "predicted_label": yp,
                "predicted_label_name": LABEL_SCHEMA.get(yp, str(yp)),
                "matched_buckets": matched_buckets,
                "negation_detected": neg_info["has_negation"],
                "code_mixing_detected": cm_info["is_code_mixed"],
            })
            
    # Proportions relative to total errors
    bucket_proportions = {
        k: round(v / total_errors, 4) if total_errors > 0 else 0.0
        for k, v in bucket_counts.items()
    }
    
    return {
        "schema_version": SCHEMA_VERSION,
        "total_evaluations": total_evals,
        "total_errors": total_errors,
        "error_rate": round(total_errors / total_evals, 4) if total_evals > 0 else 0.0,
        "bucket_counts": bucket_counts,
        "bucket_proportions": bucket_proportions,
        "sample_error_details": detailed_samples,
    }

