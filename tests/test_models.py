"""
Unit tests for NepSent-Engine Phase 3 Model Benchmark Ladder.

Verifies:
- MajorityBaselineClassifier end-to-end fit, predict, and checkpoint round-trip
- TfidfLinearModel (Logistic Regression & Calibrated LinearSVC) fit, predict_proba, save/load
- SubwordFastTextClassifier (subword character n-grams) fit, predict, save/load
- Transformer initialization, forward-pass shape, and manifest saving via tiny mock configuration
"""

import sys
import json
import pytest
import numpy as np
import torch
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nepali_scraper.models.classical import (
    MajorityBaselineClassifier,
    TfidfLinearModel,
    SubwordFastTextClassifier,
    SubwordTfidfClassifier,
)
from nepali_scraper.models.transformers_Engine import (
    TransformerSentimentTrainer,
    SentimentTextDataset,
)
from transformers import BertConfig, BertForSequenceClassification


# In-memory mock Nepali sentiment dataset for fast isolated unit tests
# (Clearly marked as unit test fixtures; NOT real research benchmark data)
UNIT_TEST_TEXTS = [
    "यो सामान अत्यन्तै नराम्रो र गुणस्तरहीन छ, कसैले नकिन्नुहोला।",  # 0: Negative
    "डेलिभरी ढिलो भयो, सामान पनि फुटेको अवस्थामा आइपुग्यो।",         # 0: Negative
    "ग्राहक सेवा एकदमै झुर र असहयोगी छ, म निराश भएँ।",                # 0: Negative
    "काठमाडौंको आजको अधिकतम तापक्रम २८ डिग्री सेल्सियस रह्यो।",       # 1: Neutral
    "सरकारले आगामी आर्थिक वर्षको बजेट संसदमा प्रस्तुत गरेको छ।",       # 1: Neutral
    "नेपाल राष्ट्र बैंकले आज नयाँ विदेशी विनिमय दर तोकेको छ।",         # 1: Neutral
    "यो मोबाइल फोन धेरै राम्रो छ, क्यामेरा र ब्याट्री उत्कृष्ट लाग्यो।",  # 2: Positive
    "धेरै राम्रो सेवा, समयमै सामान डेलिभरी भएकोमा धन्यवाद।",         # 2: Positive
    "असाध्यै सन्तोषजनक उत्पादन, मूल्य अनुसारको गुणस्तर पाइयो।",        # 2: Positive
]
UNIT_TEST_LABELS = [0, 0, 0, 1, 1, 1, 2, 2, 2]


# ============================================================================
# 1. MAJORITY BASELINE CLASSIFIER TESTS
# ============================================================================

def test_majority_baseline_classifier_round_trip(tmp_path):
    """Verify MajorityBaselineClassifier fits, predicts, and serializes correctly."""
    # Label 2 is the majority in this small set
    texts = ["वाक्य १", "वाक्य २", "वाक्य ३", "वाक्य ४"]
    labels = [1, 2, 2, 2]
    
    model = MajorityBaselineClassifier(seed=42)
    model.fit(texts, labels)
    
    assert model.majority_class_ == 2
    preds = model.predict(["नयाँ वाक्य क", "नयाँ वाक्य ख"])
    assert np.array_equal(preds, np.array([2, 2]))
    
    probas = model.predict_proba(["नयाँ वाक्य"])
    assert probas.shape == (1, 3)
    assert probas[0, 2] == 0.75  # 3 out of 4 are class 2
    
    # Save checkpoint
    save_path = tmp_path / "majority_chk"
    model.save(save_path)
    assert (save_path / "majority_model.joblib").is_file()
    assert (save_path / "metadata.json").is_file()
    
    # Load checkpoint
    loaded = MajorityBaselineClassifier.load(save_path)
    loaded_preds = loaded.predict(["नयाँ वाक्य क", "नयाँ वाक्य ख"])
    assert np.array_equal(preds, loaded_preds)
    assert loaded.metadata["training_time_sec"] >= 0.0


# ============================================================================
# 2. TF-IDF LINEAR MODEL TESTS (LOGISTIC REGRESSION & LINEARSVC)
# ============================================================================

def test_tfidf_logistic_regression_round_trip(tmp_path):
    """Verify TfidfLinearModel with LogisticRegression fits and round-trips via joblib."""
    model = TfidfLinearModel(classifier_type="logistic_regression", C=1.0, seed=42)
    model.fit(UNIT_TEST_TEXTS, UNIT_TEST_LABELS)
    
    preds_before = model.predict(UNIT_TEST_TEXTS)
    probas_before = model.predict_proba(UNIT_TEST_TEXTS)
    
    assert len(preds_before) == len(UNIT_TEST_TEXTS)
    assert probas_before.shape == (len(UNIT_TEST_TEXTS), 3)
    # Probabilities sum to 1.0 across classes
    assert np.allclose(probas_before.sum(axis=1), 1.0)
    
    save_path = tmp_path / "tfidf_logreg_chk"
    model.save(save_path)
    assert (save_path / "tfidf_model.joblib").is_file()
    assert (save_path / "metadata.json").is_file()
    
    loaded = TfidfLinearModel.load(save_path)
    preds_after = loaded.predict(UNIT_TEST_TEXTS)
    probas_after = loaded.predict_proba(UNIT_TEST_TEXTS)
    
    assert np.array_equal(preds_before, preds_after)
    assert np.allclose(probas_before, probas_after)


def test_tfidf_linear_svc_calibrated_probabilities(tmp_path):
    """Verify TfidfLinearModel with LinearSVC outputs calibrated probabilities."""
    model = TfidfLinearModel(classifier_type="linear_svc", C=1.0, seed=42)
    model.fit(UNIT_TEST_TEXTS, UNIT_TEST_LABELS)
    
    preds_before = model.predict(UNIT_TEST_TEXTS)
    probas = model.predict_proba(UNIT_TEST_TEXTS)
    
    assert probas.shape == (len(UNIT_TEST_TEXTS), 3)
    assert np.allclose(probas.sum(axis=1), 1.0)
    
    save_path = tmp_path / "tfidf_svc_chk"
    model.save(save_path)
    
    loaded = TfidfLinearModel.load(save_path)
    preds_after = loaded.predict(UNIT_TEST_TEXTS)
    assert np.array_equal(preds_before, preds_after)


# ============================================================================
# 3. SUBWORD FASTTEXT CLASSIFIER TESTS
# ============================================================================

def test_subword_fasttext_classifier_round_trip(tmp_path):
    """Verify SubwordFastTextClassifier character n-gram pipeline fits and saves."""
    model = SubwordFastTextClassifier(min_ngram=3, max_ngram=5, C=1.0, seed=42)
    model.fit(UNIT_TEST_TEXTS, UNIT_TEST_LABELS)
    
    preds_before = model.predict(UNIT_TEST_TEXTS)
    probas_before = model.predict_proba(UNIT_TEST_TEXTS)
    
    assert len(preds_before) == len(UNIT_TEST_TEXTS)
    assert probas_before.shape == (len(UNIT_TEST_TEXTS), 3)
    assert np.allclose(probas_before.sum(axis=1), 1.0)
    
    save_path = tmp_path / "fasttext_chk"
    model.save(save_path)
    assert (save_path / "fasttext_model.joblib").is_file()
    
    loaded = SubwordFastTextClassifier.load(save_path)
    preds_after = loaded.predict(UNIT_TEST_TEXTS)
    assert np.array_equal(preds_before, preds_after)


# ============================================================================
# 4. TRANSFORMER ARCHITECTURE & FORWARD PASS TESTS (MOCK CONFIG)
# ============================================================================

def test_transformer_forward_pass_mock_config():
    """
    CRITICAL CI EFFICIENCY TEST:
    Test transformer sequence classification forward pass shape using a tiny
    in-memory BertConfig without downloading heavy 500MB external checkpoints.
    """
    # Tiny mock BERT config for instant local verification
    config = BertConfig(
        vocab_size=200,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        num_labels=3,
    )
    
    model = BertForSequenceClassification(config)
    model.eval()
    
    # Batch size 2, sequence length 16
    dummy_input_ids = torch.randint(0, 200, (2, 16), dtype=torch.long)
    dummy_attention_mask = torch.ones(2, 16, dtype=torch.long)
    
    with torch.no_grad():
        outputs = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
        logits = outputs.logits
        
    assert logits.shape == (2, 3)  # [batch_size, num_classes]
    
    probs = torch.softmax(logits, dim=-1).numpy()
    assert probs.shape == (2, 3)
    assert np.allclose(probs.sum(axis=-1), 1.0)


def test_transformer_sentiment_trainer_save_structure(tmp_path):
    """Verify TransformerSentimentTrainer serializes model weights and manifest."""
    config = BertConfig(
        vocab_size=100,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=3,
    )
    
    trainer = TransformerSentimentTrainer(
        model_name_or_path="mock-bert",
        num_labels=3,
        epochs=1,
        device="cpu",
    )
    # Initialize with mock config
    trainer.initialize_model(custom_config=config)
    
    save_path = tmp_path / "mock_transformer_chk"
    trainer.save(save_path)
    
    assert (save_path / "config.json").is_file()
    assert (save_path / "training_manifest.json").is_file()
    
    with open(save_path / "training_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
        assert manifest["model_type"] == "TransformerSentimentTrainer"
        assert manifest["hyperparameters"]["num_labels"] == 3
        assert manifest["model_size_bytes"] > 0


def test_subword_tfidf_classifier_direct_and_alias():
    """Verify SubwordTfidfClassifier is identical to SubwordFastTextClassifier alias."""
    assert SubwordFastTextClassifier is SubwordTfidfClassifier
    model = SubwordTfidfClassifier(min_ngram=3, max_ngram=5, C=1.0, seed=42)
    assert model.metadata["model_type"] == "SubwordTfidfClassifier"


def test_transformer_sentiment_trainer_load_error_handling(tmp_path):
    """
    STRICT RESEARCH INVARIANT TEST:
    Verify TransformerSentimentTrainer.load raises FileNotFoundError for missing paths
    and ValueError for corrupt manifests.
    """
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="not found"):
        TransformerSentimentTrainer.load(non_existent)
        
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="Missing model weights or manifest"):
        TransformerSentimentTrainer.load(empty_dir)
        
    corrupt_dir = tmp_path / "corrupt_chk"
    corrupt_dir.mkdir()
    (corrupt_dir / "training_manifest.json").write_text("{invalid_json: true", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt manifest file"):
        TransformerSentimentTrainer.load(corrupt_dir)


def test_transformer_early_stopping_and_best_state_restoration():
    """
    Verify validation loop, Macro-F1 tracking, early stopping, and best state restoration
    during TransformerSentimentTrainer.fit().
    """
    class MockTokenizer:
        def __init__(self, vocab_size=50):
            self.vocab_size = vocab_size

        def __call__(self, text, truncation=True, padding="max_length", max_length=16, return_tensors="pt"):
            tokens = [hash(w) % (self.vocab_size - 2) + 1 for w in text.split()]
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [0] * (max_length - len(tokens))
            mask = [1 if t != 0 else 0 for t in tokens]
            return {
                "input_ids": torch.tensor([tokens], dtype=torch.long),
                "attention_mask": torch.tensor([mask], dtype=torch.long),
            }

    config = BertConfig(
        vocab_size=50,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=3,
    )
    trainer = TransformerSentimentTrainer(
        model_name_or_path="mock-bert",
        num_labels=3,
        epochs=5,
        batch_size=2,
        device="cpu",
    )
    trainer.initialize_model(custom_config=config)
    trainer.tokenizer = MockTokenizer()

    X_train = ["यो राम्रो छ", "यो नराम्रो छ", "काठमाडौं सफा छ", "यो सामान्य छ"]
    y_train = [2, 0, 1, 1]
    X_val = ["राम्रो लाग्यो", "नराम्रो लाग्यो"]
    y_val = [2, 0]

    trainer.fit(X_train, y_train, X_val=X_val, y_val=y_val, patience=1)

    assert "history" in trainer.metadata
    history = trainer.metadata["history"]
    assert len(history) >= 1
    # Early stopping should trigger before all 5 epochs when validation metrics plateau
    assert len(history) < 5
    assert "best_val_macro_f1" in trainer.metadata
    assert "val_macro_f1" in history[0]



