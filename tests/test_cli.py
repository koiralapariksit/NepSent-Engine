"""
End-to-End Unit Tests for NepSent-Engine Unified CLI and Model Inference.

Verifies the entire research pipeline via CLI subcommands:
1. `ingest`
2. `split`
3. `iaa`
4. `train`
5. `predict` (including honest calibration readout & untrained mode block)
6. `evaluate`
"""

import json
import argparse
import pytest
from pathlib import Path
from unittest.mock import patch

from nepali_scraper.cli import (
    main,
    build_parser,
    load_trained_model,
)


def test_cli_parser_build():
    """Verify that CLI parser builds with all mandatory subcommands."""
    parser = build_parser()
    subparsers_actions = [
        action for action in parser._actions 
        if isinstance(action, argparse._SubParsersAction)
    ]
    choices = subparsers_actions[0].choices
    expected_subcommands = {"ingest", "split", "iaa", "train", "evaluate", "predict"}
    assert expected_subcommands.issubset(set(choices.keys()))


def test_cli_ingest_and_split_pipeline(tmp_path):
    """Verify end-to-end ingest -> split workflow via CLI in isolated directory."""
    raw_file = tmp_path / "raw_data.jsonl"
    clean_file = tmp_path / "clean_data.jsonl"
    split_dir = tmp_path / "splits"
    
    # 1. Create mock raw records
    raw_records = [
        {"text": "काठमाडौंको मौसम आज सफा छ।", "label": 1, "domain": "news"},
        {"text": "यो मोबाइल फोन धेरै राम्रो छ, क्यामेरा उत्कृष्ट छ।", "label": 2, "domain": "reviews"},
        {"text": "सामान ढिलो आइपुग्यो, सेवा एकदमै नराम्रो छ।", "label": 0, "domain": "reviews"},
        {"text": "नेपाल राष्ट्र बैंकले आज नयाँ विदेशी विनिमय दर सार्वजनिक गर्यो।", "label": 1, "domain": "news"},
        {"text": "ग्राहक सेवाले कुनै सहयोग गरेन, म एकदम निराश भएँ।", "label": 0, "domain": "reviews"},
        {"text": "डेलिभरी समयमै भयो र सामान पनि उच्च गुणस्तरको छ।", "label": 2, "domain": "reviews"},
    ]
    with open(raw_file, "w", encoding="utf-8") as f:
        for r in raw_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    # 2. Run CLI ingest
    test_args_ingest = [
        "nepsent", "ingest",
        "--input", str(raw_file),
        "--output", str(clean_file),
        "--source", "unit_test_source",
        "--license", "CC-BY-4.0",
    ]
    with patch("sys.argv", test_args_ingest):
        exit_code = main()
        assert exit_code == 0
        
    assert clean_file.is_file()
    with open(clean_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == len(raw_records)
        
    # 3. Run CLI split
    test_args_split = [
        "nepsent", "split",
        "--input", str(clean_file),
        "--output-dir", str(split_dir),
        "--train-ratio", "0.6",
        "--val-ratio", "0.2",
        "--test-ratio", "0.2",
        "--seed", "42",
    ]
    with patch("sys.argv", test_args_split):
        exit_code = main()
        assert exit_code == 0
        
    assert (split_dir / "train.jsonl").is_file()
    assert (split_dir / "val.jsonl").is_file()
    assert (split_dir / "test.jsonl").is_file()
    assert (split_dir / "split_manifest.json").is_file()


def test_cli_iaa_workflow(tmp_path):
    """Verify CLI iaa computation between two double-blind annotator files."""
    file1 = tmp_path / "anno1.jsonl"
    file2 = tmp_path / "anno2.jsonl"
    report_file = tmp_path / "iaa_report.json"
    
    annos1 = [
        {"text": "वाक्य १", "label": 0, "domain": "news"},
        {"text": "वाक्य २", "label": 1, "domain": "news"},
        {"text": "वाक्य ३", "label": 2, "domain": "reviews"},
    ]
    annos2 = [
        {"text": "वाक्य १", "label": 0, "domain": "news"},
        {"text": "वाक्य २", "label": 1, "domain": "news"},
        {"text": "वाक्य ३", "label": 2, "domain": "reviews"},
    ]
    with open(file1, "w", encoding="utf-8") as f:
        for r in annos1:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(file2, "w", encoding="utf-8") as f:
        for r in annos2:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    test_args_iaa = [
        "nepsent", "iaa",
        "--file1", str(file1),
        "--file2", str(file2),
        "--output", str(report_file),
    ]
    with patch("sys.argv", test_args_iaa):
        exit_code = main()
        assert exit_code == 0
        
    assert report_file.is_file()
    with open(report_file, "r", encoding="utf-8") as f:
        rep = json.load(f)
        assert rep["cohen_kappa_unweighted"] == 1.0


def test_cli_train_predict_and_evaluate_lifecycle(tmp_path):
    """
    Verify complete training -> inference -> evaluation lifecycle via CLI.
    Tests model training, calibration flag checking, and bootstrap test reporting.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_file = data_dir / "train.jsonl"
    test_file = data_dir / "test.jsonl"
    chk_dir = tmp_path / "checkpoints" / "tfidf_logreg"
    report_file = tmp_path / "eval_report.json"
    
    train_records = [
        {"text": "यो सामान अत्यन्तै नराम्रो र गुणस्तरहीन छ, कसैले नकिन्नुहोला।", "label": 0},
        {"text": "डेलिभरी ढिलो भयो, सामान फुटेको छ।", "label": 0},
        {"text": "काठमाडौंको मौसम आज सफा रह्यो।", "label": 1},
        {"text": "सरकारले आज बजेट भाषण सार्वजनिक गरेको छ।", "label": 1},
        {"text": "यो मोबाइल फोन धेरै राम्रो छ, क्यामेरा उत्कृष्ट लाग्यो।", "label": 2},
        {"text": "धेरै राम्रो सेवा, समयमै सामान आइपुग्यो।", "label": 2},
    ]
    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    test_records = [
        {"text": "सामान साह्रै नराम्रो लाग्यो।", "label": 0},
        {"text": "आजको तापक्रम २२ डिग्री रह्यो।", "label": 1},
        {"text": "उत्कृष्ट सामान, पूर्ण सन्तुष्ट छु।", "label": 2},
    ]
    with open(test_file, "w", encoding="utf-8") as f:
        for r in test_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    # 1. Run CLI train
    test_args_train = [
        "nepsent", "train",
        "--data-dir", str(data_dir),
        "--model", "tfidf_logreg",
        "--output-dir", str(chk_dir),
        "--seed", "42",
    ]
    with patch("sys.argv", test_args_train):
        assert main() == 0
        
    assert (chk_dir / "tfidf_model.joblib").is_file()
    assert (chk_dir / "metadata.json").is_file()
    
    # 2. Run CLI predict on single sentence
    test_args_pred = [
        "nepsent", "predict",
        "--checkpoint", str(chk_dir),
        "--text", "यो फोन असाध्यै राम्रो छ!",
    ]
    with patch("sys.argv", test_args_pred):
        assert main() == 0
        
    # Verify loaded model properties directly
    model, cat = load_trained_model(chk_dir)
    assert cat == "tfidf"
    preds = model.predict(["यो फोन असाध्यै राम्रो छ!"])
    assert len(preds) == 1
    
    # 3. Run CLI evaluate
    test_args_eval = [
        "nepsent", "evaluate",
        "--checkpoint", str(chk_dir),
        "--test-file", str(test_file),
        "--output-report", str(report_file),
        "--n-resamples", "100",  # fast bootstrap in test
    ]
    with patch("sys.argv", test_args_eval):
        assert main() == 0
        
    assert report_file.is_file()
    with open(report_file, "r", encoding="utf-8") as f:
        eval_rep = json.load(f)
        assert "point_metrics" in eval_rep
        assert "bootstrap_confidence_intervals_95" in eval_rep
        assert "linguistic_error_taxonomy" in eval_rep


def test_cli_predict_untrained_checkpoint_blocked(tmp_path):
    """
    STRICT RESEARCH INVARIANT TEST:
    Attempting to run inference on an empty or non-existent checkpoint must fail
    with exit code 1 and block execution rather than fabricating predictions.
    """
    empty_dir = tmp_path / "empty_checkpoint"
    empty_dir.mkdir(parents=True, exist_ok=True)
    
    test_args_pred_blocked = [
        "nepsent", "predict",
        "--checkpoint", str(empty_dir),
        "--text", "कुनै पनि परीक्षण वाक्य",
    ]
    with patch("sys.argv", test_args_pred_blocked):
        exit_code = main()
        assert exit_code == 1
        
    # Test with non-existent directory
    test_args_non_existent = [
        "nepsent", "predict",
        "--checkpoint", str(tmp_path / "does_not_exist"),
        "--text", "परीक्षण",
    ]
    with patch("sys.argv", test_args_non_existent):
        exit_code = main()
        assert exit_code == 1
        
    with pytest.raises(FileNotFoundError, match="Untrained / Demo Mode Blocked"):
        load_trained_model(empty_dir)


def test_cli_ingest_zero_records_fails(tmp_path):
    """
    STRICT RESEARCH INVARIANT TEST:
    `nepsent ingest` must fail with non-zero exit code if 0 valid records are processed.
    """
    empty_file = tmp_path / "empty_input.jsonl"
    empty_file.write_text("", encoding="utf-8")
    output_file = tmp_path / "out.jsonl"
    
    test_args = [
        "nepsent", "ingest",
        "--input", str(empty_file),
        "--output", str(output_file),
    ]
    with patch("sys.argv", test_args):
        exit_code = main()
        assert exit_code != 0

