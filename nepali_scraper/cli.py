"""
NepSent-Engine Unified Command-Line Interface (CLI).

Subcommands:
  nepsent ingest    - Ingest, normalize, and validate raw text records
  nepsent split     - Leakage-free hash-grouped stratified partitioning
  nepsent iaa       - Inter-annotator agreement computation (Cohen's & Weighted Kappa)
  nepsent train     - Train classical or transformer sentiment models
  nepsent evaluate  - Test evaluation with 2,000-resample bootstrap CIs and error taxonomy
  nepsent predict   - Inference on single or batch text with calibration transparency
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

# Dynamic path resolution to ensure imports resolve when executed directly
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nepali_scraper.config import (
    PROJECT_ROOT,
    SCHEMA_VERSION,
    LABEL_SCHEMA,
    CANONICAL_DOMAINS,
    init_project_directories,
)
from nepali_scraper.processors import NepaliTextProcessor, compute_sentence_hash, normalize_nepali_unicode
from nepali_scraper.collectors import DatasetIngestor
from nepali_scraper.splitter import hash_grouped_stratified_split
from nepali_scraper.annotation import compute_iaa
from nepali_scraper.models.classical import (
    MajorityBaselineClassifier,
    TfidfLinearModel,
    SubwordTfidfClassifier,
    SubwordFastTextClassifier,
)
from nepali_scraper.evaluation.metrics import (
    compute_classification_metrics,
    bootstrap_confidence_intervals,
)
from nepali_scraper.evaluation.linguistic_analysis import (
    detect_nepali_negation,
    analyze_code_mixing,
    generate_error_taxonomy,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nepali_scraper.cli")


def load_trained_model(checkpoint_dir: Union[str, Path]) -> Tuple[Any, str]:
    """
    Load a trained sentiment model from a verified checkpoint directory.
    
    STRICT RESEARCH INVARIANT:
    Never instantiate an untrained model and pretend it is a trained classifier.
    If no checkpoint exists, raise FileNotFoundError.
    
    Args:
        checkpoint_dir: Path to model checkpoint directory
        
    Returns:
        Tuple of (loaded_model, model_category_string)
    """
    chk_path = Path(checkpoint_dir).resolve()
    if not chk_path.is_dir():
        raise FileNotFoundError(
            f"Untrained / Demo Mode Blocked — Checkpoint directory '{chk_path}' does not exist. "
            f"Train a checkpoint first using 'python -m nepali_scraper.cli train'."
        )
    
    metadata_file = chk_path / "metadata.json"
    manifest_file = chk_path / "training_manifest.json"
    
    if not metadata_file.is_file() and not manifest_file.is_file():
        raise FileNotFoundError(
            f"Untrained / Demo Mode Blocked — Checkpoint '{chk_path}' contains no model weights or metadata. "
            f"Train a checkpoint first using 'python -m nepali_scraper.cli train'."
        )
        
    model_type = ""
    if metadata_file.is_file():
        with open(metadata_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            model_type = meta.get("model_type", "")
            
    if "MajorityBaselineClassifier" in model_type or (chk_path / "majority_model.joblib").is_file():
        return MajorityBaselineClassifier.load(chk_path), "majority"
    elif "TfidfLinearModel" in model_type or (chk_path / "tfidf_model.joblib").is_file():
        return TfidfLinearModel.load(chk_path), "tfidf"
    elif (
        "SubwordTfidfClassifier" in model_type
        or "SubwordFastTextClassifier" in model_type
        or (chk_path / "subword_model.joblib").is_file()
        or (chk_path / "fasttext_model.joblib").is_file()
    ):
        return SubwordTfidfClassifier.load(chk_path), "subword_tfidf"
    elif manifest_file.is_file() or (chk_path / "config.json").is_file():
        from nepali_scraper.models.transformers_Engine import TransformerSentimentTrainer
        trainer = TransformerSentimentTrainer.load(chk_path)
        return trainer, "transformer"
    else:
        raise FileNotFoundError(
            f"Untrained / Demo Mode Blocked — Unrecognized checkpoint structure at '{chk_path}'."
        )


# ============================================================================
# SUBCOMMAND HANDLERS
# ============================================================================

def handle_ingest(args: argparse.Namespace) -> int:
    """Handle 'ingest' subcommand."""
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    
    if not input_path.is_file():
        logger.error(f"Input file not found: {input_path}")
        return 1
        
    ingestor = DatasetIngestor()
    
    # Load raw records
    raw_records = []
    if input_path.suffix.lower() == ".jsonl":
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_records.append(json.loads(line))
    elif input_path.suffix.lower() == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            raw_records = json.load(f)
    elif input_path.suffix.lower() == ".csv":
        import csv
        with open(input_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            raw_records = list(reader)
    else:
        # Assume plain text lines
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_records.append({"text": line.strip()})
                    
    logger.info(f"Loaded {len(raw_records)} raw records from {input_path}")
    for r in raw_records:
        r.setdefault("source", args.source)
        r.setdefault("license", args.license)
        r.setdefault("domain", args.domain)
        r.setdefault("provenance_status", "UNVERIFIED")
        r.setdefault("collection_timestamp", None)
        
    clean_records = ingestor.ingest_records(raw_records)
    stats = ingestor.stats
    
    if len(clean_records) == 0:
        logger.error(f"Ingestion failed: 0 valid records processed from {input_path}")
        return 1
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in clean_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    logger.info(f"Ingested {len(clean_records)} clean records -> {output_path}")
    logger.info(f"Ingestion stats: {json.dumps(stats, ensure_ascii=False, indent=2)}")
    return 0


def handle_split(args: argparse.Namespace) -> int:
    """Handle 'split' subcommand."""
    input_path = Path(args.input).resolve()
    out_dir = Path(args.output_dir).resolve()
    
    if not input_path.is_file():
        logger.error(f"Input clean JSONL not found: {input_path}")
        return 1
        
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    logger.info(f"Loaded {len(records)} records for hash-grouped splitting.")
    split_res = hash_grouped_stratified_split(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    
    out_dir.mkdir(parents=True, exist_ok=True)
    for partition_name in ["train", "val", "test"]:
        part_records = split_res[partition_name]
        part_file = out_dir / f"{partition_name}.jsonl"
        with open(part_file, "w", encoding="utf-8") as f:
            for r in part_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
    manifest = split_res.get("metadata", split_res.get("manifest", {}))
    manifest_file = out_dir / "split_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Splits saved to {out_dir}")
    logger.info(f"Manifest: {json.dumps(manifest, ensure_ascii=False, indent=2)}")
    return 0


def handle_iaa(args: argparse.Namespace) -> int:
    """Handle 'iaa' subcommand with stable-ID alignment and Unicode validation."""
    file1 = Path(args.file1).resolve()
    file2 = Path(args.file2).resolve()
    
    def _read_annos_dict(path: Path) -> Dict[str, Dict[str, Any]]:
        lookup = {}
        if path.suffix.lower() == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    if line.strip():
                        r = json.loads(line)
                        rec_id = r.get("id") or r.get("record_id") or r.get("sha256_hash")
                        if not rec_id:
                            t = r.get(args.text_key, "")
                            rec_id = compute_sentence_hash(t) if t else f"line_{line_no}"
                        lookup[rec_id] = r
        else:
            import csv
            with open(path, "r", encoding="utf-8") as f:
                for line_no, r in enumerate(csv.DictReader(f), start=1):
                    rec_id = r.get("id") or r.get("record_id") or r.get("sha256_hash")
                    if not rec_id:
                        t = r.get(args.text_key, "")
                        rec_id = compute_sentence_hash(t) if t else f"line_{line_no}"
                    lookup[rec_id] = r
        return lookup
        
    annos1_dict = _read_annos_dict(file1)
    annos2_dict = _read_annos_dict(file2)
    
    # Invariant C1: Stable-ID Alignment validation
    ids1 = set(annos1_dict.keys())
    ids2 = set(annos2_dict.keys())
    if ids1 != ids2:
        raise ValueError(
            f"Annotation ID mismatch: {len(ids1 - ids2)} records missing in file2, "
            f"{len(ids2 - ids1)} records missing in file1."
        )
        
    common_ids = sorted(list(ids1))
    labels_a = []
    labels_b = []
    domains = []
    
    for rid in common_ids:
        r1 = annos1_dict[rid]
        r2 = annos2_dict[rid]
        t1 = normalize_nepali_unicode(r1.get(args.text_key, ""))
        t2 = normalize_nepali_unicode(r2.get(args.text_key, ""))
        if t1 != t2:
            raise ValueError(
                f"Annotation text mismatch for stable ID '{rid}':\n"
                f"File 1: {t1}\n"
                f"File 2: {t2}"
            )
        labels_a.append(int(r1[args.label_key]))
        labels_b.append(int(r2[args.label_key]))
        if args.domain_key in r1:
            domains.append(r1[args.domain_key])
            
    iaa_report = compute_iaa(
        labels_a=labels_a,
        labels_b=labels_b,
        domains=domains if domains else None,
    )
    
    out_str = json.dumps(iaa_report, ensure_ascii=False, indent=2)
    print(out_str)
    
    if args.output:
        out_p = Path(args.output).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(out_str)
        logger.info(f"IAA report saved to {out_p}")
    return 0


def handle_train(args: argparse.Namespace) -> int:
    """Handle 'train' subcommand."""
    data_dir = Path(args.data_dir).resolve()
    train_file = data_dir / "train.jsonl"
    out_dir = Path(args.output_dir).resolve()
    
    if not train_file.is_file():
        logger.error(f"Train split not found at {train_file}")
        return 1
        
    train_texts = []
    train_labels = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                train_texts.append(item["text"])
                train_labels.append(int(item["label"]))
                
    logger.info(f"Loaded {len(train_texts)} training samples.")
    model_choice = args.model.lower()
    
    if model_choice == "majority":
        model = MajorityBaselineClassifier(seed=args.seed)
    elif model_choice == "tfidf_logreg":
        model = TfidfLinearModel(classifier_type="logistic_regression", seed=args.seed)
    elif model_choice in ["tfidf_svm", "tfidf_linearsvc"]:
        model = TfidfLinearModel(classifier_type="linear_svc", seed=args.seed)
    elif model_choice in ["subword_tfidf", "fasttext"]:
        model = SubwordTfidfClassifier(seed=args.seed)
    elif model_choice in ["mbert", "muril", "nepberta"]:
        from nepali_scraper.models.transformers_Engine import TransformerSentimentTrainer
        val_file = data_dir / "val.jsonl"
        val_texts, val_labels = None, None
        if val_file.is_file():
            val_texts, val_labels = [], []
            with open(val_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        val_texts.append(item["text"])
                        val_labels.append(int(item["label"]))
        trainer = TransformerSentimentTrainer(model_name_or_path=model_choice, seed=args.seed)
        logger.info(f"Fine-tuning transformer: {model_choice}...")
        trainer.fit(train_texts, train_labels, X_val=val_texts, y_val=val_labels)
        out_dir.mkdir(parents=True, exist_ok=True)
        trainer.save(out_dir)
        logger.info(f"Model saved successfully to {out_dir}")
        return 0
    else:
        logger.error(f"Unsupported model: {model_choice}")
        return 1
        
    logger.info(f"Fitting model: {model_choice}...")
    model.fit(train_texts, train_labels)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir)
    logger.info(f"Model saved successfully to {out_dir}")
    return 0


def handle_evaluate(args: argparse.Namespace) -> int:
    """Handle 'evaluate' subcommand."""
    chk_dir = Path(args.checkpoint).resolve()
    test_file = Path(args.test_file).resolve()
    
    if not test_file.is_file():
        logger.error(f"Test file not found: {test_file}")
        return 1
        
    try:
        model, model_cat = load_trained_model(chk_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
        
    test_records = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_records.append(json.loads(line))
                
    test_texts = [r["text"] for r in test_records]
    test_labels = [int(r["label"]) for r in test_records]
    
    predictions = model.predict(test_texts)
    
    # 1. Point metrics
    metrics = compute_classification_metrics(test_labels, predictions)
    
    # 2. Non-parametric bootstrap CIs (B=2,000 resamples)
    boot_ci = bootstrap_confidence_intervals(
        test_labels,
        predictions,
        n_resamples=args.n_resamples,
        ci=0.95,
        seed=args.seed,
    )
    
    # 3. Linguistic Error Taxonomy
    taxonomy = generate_error_taxonomy(test_records, test_labels, predictions)
    
    report = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": str(chk_dir),
        "test_samples": len(test_labels),
        "point_metrics": metrics,
        "bootstrap_confidence_intervals_95": boot_ci,
        "linguistic_error_taxonomy": taxonomy,
    }
    
    out_str = json.dumps(report, ensure_ascii=False, indent=2)
    print(out_str)
    
    if args.output_report:
        out_p = Path(args.output_report).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(out_str)
        logger.info(f"Evaluation report written to {out_p}")
        
    return 0


def handle_predict(args: argparse.Namespace) -> int:
    """
    Handle 'predict' subcommand.
    
    STRICT RULES:
    1. Blocks execution if checkpoint is missing or untrained.
    2. Explicitly flags probability calibration status.
    """
    chk_dir = Path(args.checkpoint).resolve()
    
    try:
        model, model_cat = load_trained_model(chk_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.stderr.write(f"ERROR: {str(e)}\n")
        return 1
        
    # Gather input texts
    input_texts = []
    if args.text:
        input_texts.append(args.text)
    elif args.input:
        in_p = Path(args.input).resolve()
        if in_p.suffix.lower() == ".jsonl":
            with open(in_p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        input_texts.append(json.loads(line).get("text", line.strip()))
        else:
            with open(in_p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        input_texts.append(line.strip())
    else:
        logger.error("Must provide either --text '<nepali_text>' or --input <file>")
        return 1
        
    predictions = model.predict(input_texts)
    
    # Check probability calibration
    has_proba = hasattr(model, "predict_proba")
    probas = model.predict_proba(input_texts) if has_proba else None
    
    # Determine honest calibration status
    is_calibrated = False
    calibration_status = "Uncalibrated Softmax Score (Heuristic Only)"
    
    if model_cat == "tfidf":
        is_calibrated = True
        calibration_status = "Platt-Calibrated Probability"
    elif model_cat == "majority":
        is_calibrated = True
        calibration_status = "Empirical Training Class Prior"
        
    results = []
    for idx, text in enumerate(input_texts):
        pred_label = int(predictions[idx])
        pred_name = LABEL_SCHEMA.get(pred_label, str(pred_label))
        
        confidence = float(np.max(probas[idx])) if probas is not None else 1.0
        dist = {
            LABEL_SCHEMA.get(c, str(c)): float(round(probas[idx, c], 4))
            for c in [0, 1, 2]
        } if probas is not None else {}
        
        # Real-time linguistic readout
        neg_info = detect_nepali_negation(text)
        cm_info = analyze_code_mixing(text)
        
        results.append({
            "text": text,
            "predicted_label": pred_label,
            "sentiment": pred_name,
            "confidence_score": round(confidence, 4),
            "is_calibrated": is_calibrated,
            "calibration_status": calibration_status,
            "probability_distribution": dist,
            "linguistic_readout": {
                "has_negation": neg_info["has_negation"],
                "negation_tokens": neg_info["matched_tokens"],
                "is_code_mixed": cm_info["is_code_mixed"],
                "devanagari_ratio": cm_info["devanagari_ratio"],
            }
        })
        
    out_str = json.dumps(results if len(results) > 1 else results[0], ensure_ascii=False, indent=2)
    print(out_str)
    
    if args.output:
        out_p = Path(args.output).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(out_str)
            
    return 0


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build root CLI parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="nepsent",
        description="NepSent-Engine: Scientifically Valid Nepali Sentiment Analysis CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")
    
    # 1. ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest and validate raw text records")
    p_ingest.add_argument("--input", required=True, help="Input raw data file (CSV, JSONL, JSON, TXT)")
    p_ingest.add_argument("--output", required=True, help="Output clean JSONL file path")
    p_ingest.add_argument("--source", default="UNVERIFIED", help="Source provenance identifier")
    p_ingest.add_argument("--license", default="UNKNOWN_REVIEW_REQUIRED", help="Data license")
    p_ingest.add_argument("--domain", default="general_unclassified", choices=list(CANONICAL_DOMAINS), help="Domain identifier")
    
    # 2. split
    p_split = subparsers.add_parser("split", help="Leakage-free hash-grouped stratified partition")
    p_split.add_argument("--input", required=True, help="Clean JSONL dataset")
    p_split.add_argument("--output-dir", required=True, help="Directory to save train/val/test splits")
    p_split.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio (default: 0.8)")
    p_split.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio (default: 0.1)")
    p_split.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio (default: 0.1)")
    p_split.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    
    # 3. iaa
    p_iaa = subparsers.add_parser("iaa", help="Compute Inter-Annotator Agreement (Cohen's & Weighted Kappa)")
    p_iaa.add_argument("--file1", required=True, help="Annotator 1 file (JSONL/CSV)")
    p_iaa.add_argument("--file2", required=True, help="Annotator 2 file (JSONL/CSV)")
    p_iaa.add_argument("--text-key", default="text", help="Text column key")
    p_iaa.add_argument("--label-key", default="label", help="Label column key")
    p_iaa.add_argument("--domain-key", default="domain", help="Domain column key")
    p_iaa.add_argument("--output", help="Optional output JSON path")
    
    # 4. train
    p_train = subparsers.add_parser("train", help="Train classical or transformer sentiment models")
    p_train.add_argument("--data-dir", required=True, help="Directory containing train.jsonl")
    p_train.add_argument("--model", required=True, choices=[
        "majority", "tfidf_logreg", "tfidf_svm", "subword_tfidf", "fasttext", "mbert", "muril", "nepberta"
    ], help="Model architecture choice")
    p_train.add_argument("--output-dir", required=True, help="Directory to save model checkpoint")
    p_train.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    
    # 5. evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run evaluation with bootstrap CIs and error taxonomy")
    p_eval.add_argument("--checkpoint", required=True, help="Trained model checkpoint directory")
    p_eval.add_argument("--test-file", required=True, help="Test JSONL split")
    p_eval.add_argument("--output-report", help="Output JSON report path")
    p_eval.add_argument("--n-resamples", type=int, default=2000, help="Bootstrap resamples (default: 2,000)")
    p_eval.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    
    # 6. predict
    p_pred = subparsers.add_parser("predict", help="Predict sentiment for single text or batch file")
    p_pred.add_argument("--checkpoint", required=True, help="Trained model checkpoint directory")
    p_pred.add_argument("--text", help="Single Nepali sentence string to classify")
    p_pred.add_argument("--input", help="Batch input file (JSONL/TXT)")
    p_pred.add_argument("--output", help="Optional output JSON file")
    
    return parser


def main() -> int:
    """CLI execution entrypoint."""
    init_project_directories()
    parser = build_parser()
    args = parser.parse_args()
    
    if not args.subcommand:
        parser.print_help()
        return 0
        
    handlers = {
        "ingest": handle_ingest,
        "split": handle_split,
        "iaa": handle_iaa,
        "train": handle_train,
        "evaluate": handle_evaluate,
        "predict": handle_predict,
    }
    
    handler = handlers.get(args.subcommand)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
