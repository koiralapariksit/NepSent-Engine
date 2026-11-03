"""
Unit Tests for Phase 4: Statistical Evaluation, LODO Generalization,
Subword Fertility, and Nepali Linguistic Error Taxonomy.
"""

import pytest
import numpy as np
from nepali_scraper.evaluation.metrics import (
    compute_classification_metrics,
    bootstrap_confidence_intervals,
    paired_bootstrap_test,
    mcnemar_test,
)
from nepali_scraper.evaluation.cross_domain import (
    evaluate_per_domain,
    run_lodo_evaluation,
)
from nepali_scraper.evaluation.linguistic_analysis import (
    detect_nepali_negation,
    analyze_code_mixing,
    compute_subword_fertility,
    generate_error_taxonomy,
)
from nepali_scraper.models.classical import MajorityBaselineClassifier


# ============================================================================
# 1. STATISTICAL EVALUATION & BOOTSTRAP CI TESTS
# ============================================================================

def test_bootstrap_confidence_intervals_bounds():
    """Verify that empirical percentile bootstrap CI bounds are valid: lower <= point <= upper."""
    # Synthetic balanced ground truth and slightly imperfect predictions
    y_true = np.array([0] * 30 + [1] * 30 + [2] * 30)
    y_pred = np.array([0] * 25 + [1] * 5 + [1] * 25 + [2] * 5 + [2] * 25 + [0] * 5)
    
    ci_res = bootstrap_confidence_intervals(y_true, y_pred, n_resamples=500, ci=0.95, seed=42)
    
    # Check Accuracy bounds
    acc = ci_res["accuracy"]
    assert acc["ci_lower"] <= acc["point"] <= acc["ci_upper"], f"Invalid Acc bounds: {acc}"
    assert 0.0 <= acc["ci_lower"] <= 1.0
    assert 0.0 <= acc["ci_upper"] <= 1.0
    
    # Check Macro F1 bounds
    macro = ci_res["macro_f1"]
    assert macro["ci_lower"] <= macro["point"] <= macro["ci_upper"], f"Invalid Macro F1 bounds: {macro}"
    assert 0.0 <= macro["ci_lower"] <= 1.0
    assert 0.0 <= macro["ci_upper"] <= 1.0
    
    # Check per-class F1 bounds
    for lbl in [0, 1, 2]:
        cls_ci = ci_res["per_class_f1"][lbl]
        assert cls_ci["ci_lower"] <= cls_ci["point"] <= cls_ci["ci_upper"], f"Invalid Class {lbl} bounds: {cls_ci}"


def test_paired_bootstrap_hypothesis_testing():
    """Verify paired bootstrap hypothesis testing with identical vs divergent models."""
    y_true = np.array([0, 0, 1, 1, 2, 2] * 10)
    
    # Identical predictions -> delta = 0, p-value = 1.0
    pred_a = y_true.copy()
    pred_b = y_true.copy()
    test_ident = paired_bootstrap_test(y_true, pred_a, pred_b, n_resamples=200, seed=42)
    assert test_ident["delta_obs"] == 0.0
    assert test_ident["p_value"] == 1.0
    assert test_ident["significant_at_05"] is False
    
    # Model A is perfect, Model B has random/poor predictions
    rng = np.random.default_rng(42)
    pred_bad = rng.choice([0, 1, 2], size=len(y_true))
    test_diff = paired_bootstrap_test(y_true, pred_a, pred_bad, n_resamples=500, seed=42)
    assert test_diff["delta_obs"] > 0.4
    assert test_diff["p_value"] < 0.05
    assert test_diff["significant_at_05"] is True


def test_mcnemar_test_exact_and_chi2():
    """Verify McNemar contingency testing under small (binomial) and large (chi2) sample sizes."""
    # 1. Exact binomial branch (discordant < 25)
    y_true_small = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    pred_a_small = np.array([0, 1, 2, 0, 1, 2, 0, 1])  # 8/8 correct
    pred_b_small = np.array([1, 2, 0, 1, 2, 0, 1, 2])  # 0/8 correct
    res_small = mcnemar_test(y_true_small, pred_a_small, pred_b_small)
    assert res_small["method"] == "exact_binomial"
    assert res_small["b"] == 8
    assert res_small["c"] == 0
    assert res_small["p_value"] < 0.05
    assert res_small["significant_at_05"] is True
    
    # 2. Large sample branch (discordant >= 25)
    y_true_large = np.zeros(100, dtype=int)
    pred_a_large = np.zeros(100, dtype=int)  # all 100 correct
    pred_b_large = np.ones(100, dtype=int)   # all 100 incorrect
    res_large = mcnemar_test(y_true_large, pred_a_large, pred_b_large)
    assert res_large["method"] == "continuity_corrected_chi2"
    assert res_large["b"] == 100
    assert res_large["c"] == 0
    assert res_large["p_value"] < 0.001


# ============================================================================
# 2. NEPALI MORPHOLOGICAL NEGATION DETECTOR TESTS
# ============================================================================

def test_nepali_negation_detector_flags_authentic_negation():
    """Verify genuine Nepali negation (copulas, suffixes, prefixes) is correctly detected."""
    positive_negation_cases = [
        ("यो काम राम्रो छैन।", ["छैन"]),
        ("म त्यहाँ जान्नँ।", ["जान्नँ"]),
        ("उसले काम गरेन।", ["गरेन"]),
        ("यो उत्पादन किन्नु पर्दैन।", ["पर्दैन"]),
        ("सार्वजनिक स्थलमा फोहोर फाल्नु हुँदैन।", ["हुँदैन"]),
        ("उनले आज भात खाइनन्।", ["खाइनन्"]),
        ("यो सामान कसैले नकिन्नुहोला।", ["नकिन्नुहोला"]),
        ("तपाईं त्यहाँ नजानुहोला।", ["नजानुहोला"]),
        ("त्यसो न गर।", ["न"]),
    ]
    
    for sentence, expected_tokens in positive_negation_cases:
        res = detect_nepali_negation(sentence)
        assert res["has_negation"] is True, f"Failed to detect negation in: '{sentence}'"
        for tok in expected_tokens:
            assert tok in res["matched_tokens"], f"Expected token '{tok}' not in {res['matched_tokens']}"


def test_nepali_negation_detector_does_not_false_fire_on_nouns_with_na():
    """
    CRITICAL LINGUISTIC INVARIANT:
    Words starting with letter 'न' or containing non-negative suffixes
    MUST NOT be falsely flagged as negation (e.g. नेपाल, नयाँ, नदी, नागरिक, नम्र).
    """
    non_negative_cases = [
        "नेपाल एक सुन्दर र शान्त देश हो।",
        "नयाँ किताब धेरै राम्रो छ।",
        "नदी किनारमा हरिया रुखहरू छन्।",
        "नागरिक समाजको भूमिका महत्वपूर्ण छ।",
        "नम्र बोली सबैलाई मन पर्छ।",
        "नेताहरूले राष्ट्रिय एकताको भाषण गरे।",
        "यो नयाँ नियम लागू भएको छ।",
        "हामी प्लेन चढेर पोखरा गयौं।",
        "विज्ञान र प्रविधिको विकास तीव्र छ।",
        "उहाँ एक विद्वान व्यक्ति हुनुहुन्छ।",
    ]
    
    for sentence in non_negative_cases:
        res = detect_nepali_negation(sentence)
        assert res["has_negation"] is False, (
            f"False positive negation detected in non-negative text: '{sentence}'. "
            f"Matched tokens: {res['matched_tokens']}"
        )


# ============================================================================
# 3. CODE-MIXING & SCRIPT ANALYZER TESTS
# ============================================================================

def test_code_mixing_analyzer():
    """Verify detection of Latin/English tokens and Devanagari code-mixing."""
    # Pure Devanagari
    pure_dev = "काठमाडौंको मौसम आज निकै सफा छ।"
    res_pure = analyze_code_mixing(pure_dev)
    assert res_pure["is_code_mixed"] is False
    assert res_pure["latin_chars"] == 0
    assert len(res_pure["latin_tokens"]) == 0
    assert res_pure["devanagari_ratio"] > 0.8
    
    # Code-mixed Nepali + English words
    mixed = "यो product साह्रै राम्रो लाग्यो, delivery पनि super fast थियो!"
    res_mixed = analyze_code_mixing(mixed)
    assert res_mixed["is_code_mixed"] is True
    assert set(res_mixed["latin_tokens"]) == {"product", "delivery", "super", "fast"}
    assert res_mixed["latin_ratio"] > 0.3
    assert res_mixed["devanagari_ratio"] > 0.3


# ============================================================================
# 4. SUBWORD FERTILITY CALCULATOR TESTS
# ============================================================================

class MockTokenizer:
    """Mock tokenizer simulating subword fragmentation."""
    def tokenize(self, text: str):
        # Splits every word into 2 tokens, except words containing 'अति' which get 3
        words = text.split()
        tokens = []
        for w in words:
            if "अति" in w:
                tokens.extend([w[:2], w[2:4], w[4:]])
            elif "अज्ञात" in w:
                tokens.append("[UNK]")
            else:
                tokens.extend([w[:len(w)//2], w[len(w)//2:]])
        return tokens


def test_subword_fertility_calculator():
    """Verify subword fertility, character-to-token ratio, and UNK calculation."""
    tokenizer = MockTokenizer()
    texts = [
        "यो पुस्तक राम्रो छ",  # 4 words
        "अति सुन्दर रचना अज्ञात",  # 4 words, one with अति (3 tokens), one UNK
    ]
    
    fert_res = compute_subword_fertility(texts, tokenizer)
    assert fert_res["texts_analyzed"] == 2
    assert fert_res["total_words"] == 8
    assert fert_res["fertility"] > 1.5  # Subword tokens per word > 1.0
    assert fert_res["unk_count"] == 1
    assert fert_res["unk_rate"] > 0.0


# ============================================================================
# 5. CROSS-DOMAIN & LODO EVALUATION TESTS
# ============================================================================

def test_evaluate_per_domain_and_power_warning():
    """Verify domain metrics breakdown and statistical power warning for N < 50."""
    y_true = [0, 1, 2] * 20  # 60 samples
    y_pred = [0, 1, 2] * 20
    # 55 in 'news', 5 in 'social' (triggers exploratory warning)
    domains = ["news"] * 55 + ["social"] * 5
    
    res = evaluate_per_domain(y_true, y_pred, domains)
    assert "news" in res["domain_breakdown"]
    assert "social" in res["domain_breakdown"]
    
    # 'news' has N=55 >= 50
    assert res["domain_breakdown"]["news"]["is_exploratory"] is False
    assert res["domain_breakdown"]["news"]["warning"] is None
    
    # 'social' has N=5 < 50 -> flagged as exploratory
    assert res["domain_breakdown"]["social"]["is_exploratory"] is True
    assert "N=5 < 50" in res["domain_breakdown"]["social"]["warning"]


def test_run_lodo_evaluation_generalization():
    """Verify Leave-One-Domain-Out training and evaluation workflow across 3 domains."""
    # Synthetic records across 3 domains
    records = [
        {"text": "समाचार १ राम्रो", "label": 2, "domain": "news"},
        {"text": "समाचार २ नराम्रो", "label": 0, "domain": "news"},
        {"text": "समीक्षा १ राम्रो", "label": 2, "domain": "reviews"},
        {"text": "समीक्षा २ सामान्य", "label": 1, "domain": "reviews"},
        {"text": "ट्विट १ सामान्य", "label": 1, "domain": "social"},
        {"text": "ट्विट २ नराम्रो", "label": 0, "domain": "social"},
    ]
    
    # Model factory instantiating MajorityBaselineClassifier
    def model_factory():
        return MajorityBaselineClassifier(seed=42)
    
    lodo_res = run_lodo_evaluation(records, model_factory)
    assert lodo_res["n_domains"] == 3
    assert "news" in lodo_res["per_domain_results"]
    assert "reviews" in lodo_res["per_domain_results"]
    assert "social" in lodo_res["per_domain_results"]
    assert 0.0 <= lodo_res["mean_lodo_macro_f1"] <= 1.0


# ============================================================================
# 6. ERROR TAXONOMY GENERATOR TESTS
# ============================================================================

def test_error_taxonomy_categorization():
    """Verify categorization of false predictions into diagnostic error buckets."""
    records = [
        {"text": "यो सामान राम्रो छैन।"},  # Has negation: True
        {"text": "मौसम सामान्य रह्यो।"},   # Neutral text
        {"text": "यो product राम्रो छ, Best service in town!"}, # Code-mixed
    ]
    y_true = [0, 1, 2]
    # Predict 2 for record 0 (Negation error), 0 for record 1 (Neutral-Polar confusion), 1 for record 2 (Code-mixed)
    y_pred = [2, 0, 1]
    
    taxonomy = generate_error_taxonomy(records, y_true, y_pred)
    assert taxonomy["total_evaluations"] == 3
    assert taxonomy["total_errors"] == 3
    assert taxonomy["error_rate"] == 1.0
    
    counts = taxonomy["bucket_counts"]
    assert counts["negation_scope_error"] >= 1
    assert counts["neutral_polar_boundary_confusion"] >= 1
    assert counts["code_mixed_error"] >= 1

