"""
Phase 3 Orchestrator - Model Training & Benchmarking Coordinator

Runs the complete 4-tier model ladder:
1. TF-IDF + LogReg (55-65% F1)
2. fastText (63-72% F1)
3. mBERT (74-82% F1)
4. MuRIL (79-86% F1)

With cross-validation and cross-domain evaluation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def show_banner():
    """Display Phase 3 banner."""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║        NEPALI SENTIMENT ANALYSIS - PHASE 3 ORCHESTRATOR                  ║
║        Model Training, Benchmarking & 4-Tier Ladder Evaluation           ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
""")


def show_methodology():
    """Display methodology."""
    print("""
PHASE 3: MODEL BENCHMARKING METHODOLOGY
─────────────────────────────────────────────────────────────────────────

4-TIER MODEL LADDER:

Level 1: TF-IDF + Logistic Regression
  └─ Features: Char-level trigrams (2000 features)
  └─ Classifier: Logistic Regression (max_iter=1000)
  └─ Target: 55–65% Macro F1
  └─ Speed: ~10 seconds

Level 2: fastText Embeddings + Logistic Regression
  └─ Features: 300-dim pre-trained Nepali embeddings
  └─ Classifier: Logistic Regression
  └─ Target: 63–72% Macro F1
  └─ Speed: ~1 minute

Level 3: Fine-tuned mBERT
  └─ Model: bert-base-multilingual-uncased
  └─ Training: 3 epochs, batch_size=32, lr=2e-5
  └─ Target: 74–82% Macro F1
  └─ Speed: ~10 minutes

Level 4: Fine-tuned MuRIL (BEST)
  └─ Model: google/muril-base-cased (Indic-optimized)
  └─ Training: 3 epochs, batch_size=32, lr=5e-5
  └─ Target: 79–86% Macro F1
  └─ Speed: ~15 minutes


EVALUATION PROTOCOL:

Train/Test Split:
  • 85% training (93 sentences)
  • 15% test (16 sentences)
  • Stratified by label distribution
  • Random seed: 42

Cross-Validation:
  • 5-fold stratified (Levels 1-2 only)
  • Ensures generalization
  • Levels 3-4 use single split for speed

Cross-Domain Transfer:
  • Train: news_politics only
  • Test: e-commerce + social_media only
  • Measures model robustness
  • Key metric for publication


METRICS:

Primary:
  • Macro F1: Average precision & recall across all classes
  • Accuracy: (TP + TN) / Total

Secondary:
  • Per-class Precision, Recall, F1
  • Confusion Matrix
  • Classification Report


OUTPUT FILES:

Per Model:
  • 08a_tfidf_results.json (TF-IDF metrics)
  • 08b_fasttext_results.json (fastText metrics)
  • 08c_mbert_results.json (mBERT metrics)
  • 08d_muril_results.json (MuRIL metrics - BEST)

Summary:
  • 09_model_comparison.json (Rankings & analysis)
  • 09_error_analysis.json (Failure categorization)


EXPECTED PERFORMANCE:

Baseline (Random): 33.3% macro F1
Level 1 (TF-IDF): 55-65%
Level 2 (fastText): 63-72%
Level 3 (mBERT): 74-82%
Level 4 (MuRIL): 79-86% ← BEST for Nepali


HARDWARE REQUIREMENTS:

CPU-only:
  • TF-IDF: Very fast (~10 sec)
  • fastText: ~1 min
  • mBERT: ~20 min (slower on CPU)
  • MuRIL: ~25 min

GPU (recommended):
  • mBERT: ~3-5 min
  • MuRIL: ~5-7 min

""")


def run_phase3_orchestrator():
    """Run complete Phase 3."""
    show_banner()
    show_methodology()
    
    print("="*80)
    print(" PHASE 3 EXECUTION")
    print("="*80)
    print()
    
    dataset_file = Path("d:/ds mdoel/data/04_annotated_dataset.jsonl")
    
    if not dataset_file.exists():
        print(f"✗ Dataset not found: {dataset_file}")
        print("  Please complete Phase 2 first")
        return False
    
    print("Phase 3 Models and Scripts:")
    print("  • phase3_model_trainer.py (base trainer class)")
    print("  • phase3_tfidf_baseline.py (Level 1)")
    print("  • phase3_fasttext_model.py (Level 2)")
    print("  • phase3_mbert_finetuning.py (Level 3)")
    print("  • phase3_muril_finetuning.py (Level 4 - BEST)")
    print()
    
    print("Training Schedule:")
    print("  [1/4] TF-IDF Baseline          ... ~30 sec")
    print("  [2/4] fastText Embeddings      ... ~2 min")
    print("  [3/4] Fine-tuned mBERT         ... ~20 min")
    print("  [4/4] Fine-tuned MuRIL (BEST)  ... ~25 min")
    print("        ─────────────────────────────────────")
    print("        Total:                     ~45 min")
    print()
    
    print("To run individual models:")
    print("  $ python phase3_tfidf_baseline.py      # Level 1")
    print("  $ python phase3_fasttext_model.py      # Level 2")
    print("  $ python phase3_mbert_finetuning.py    # Level 3")
    print("  $ python phase3_muril_finetuning.py    # Level 4")
    print()
    
    print("To run all models sequentially:")
    print("  $ python phase3_orchestrator.py --run-all")
    print()
    
    print("Output Analysis:")
    print("  $ python phase3_error_analysis.py")
    print()
    
    print("="*80)
    print(" READY FOR TRAINING")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Review phase3_tfidf_baseline.py for example structure")
    print("  2. Run: python phase3_tfidf_baseline.py")
    print("  3. Monitor each level's performance")
    print("  4. Compare results in 09_model_comparison.json")
    print()


def run_all_models():
    """Run all 4 models sequentially."""
    print("\n" + "="*80)
    print(" RUNNING ALL 4 MODELS")
    print("="*80 + "\n")
    
    models = [
        ("TF-IDF Baseline", "phase3_tfidf_baseline"),
        ("fastText Model", "phase3_fasttext_model"),
        ("mBERT Fine-tuning", "phase3_mbert_finetuning"),
        ("MuRIL Fine-tuning", "phase3_muril_finetuning"),
    ]
    
    results = {}
    
    for i, (name, module) in enumerate(models, 1):
        print(f"\n[{i}/4] {name}")
        print("-" * 80)
        
        try:
            # Dynamically import and run
            if module == "phase3_tfidf_baseline":
                from phase3_tfidf_baseline import main
            elif module == "phase3_fasttext_model":
                from phase3_fasttext_model import main
            elif module == "phase3_mbert_finetuning":
                from phase3_mbert_finetuning import main
            elif module == "phase3_muril_finetuning":
                from phase3_muril_finetuning import main
            
            main()
            results[name] = "✓ Completed"
        except Exception as e:
            print(f"\n✗ Error in {name}: {e}")
            results[name] = f"✗ Failed: {str(e)}"
    
    # Summary
    print("\n" + "="*80)
    print(" PHASE 3 SUMMARY")
    print("="*80 + "\n")
    
    for i, (name, status) in enumerate(results.items(), 1):
        print(f"{i}. {name:30s} {status}")
    
    print()
    print("Results saved to:")
    print("  • 08a_tfidf_results.json")
    print("  • 08b_fasttext_results.json")
    print("  • 08c_mbert_results.json")
    print("  • 08d_muril_results.json")
    print("  • 09_model_comparison.json")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Phase 3 Orchestrator - Model Training & Benchmarking"
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all 4 models sequentially"
    )
    
    args = parser.parse_args()
    
    if args.run_all:
        run_all_models()
    else:
        run_phase3_orchestrator()
