"""
PHASE 2 ORCHESTRATOR - Human-In-The-Loop Annotation Workflow

Coordinates:
1. Data splitting (validation + training)
2. Annotation (human or simulated)
3. Cohen's Kappa calculation
4. Consensus mediation (if needed)
5. Final annotation merge
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║          NEPALI SENTIMENT ANALYSIS - PHASE 2 ORCHESTRATOR                ║
║          Human-In-The-Loop Annotation & Mathematical Validation           ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

def show_banner():
    """Display phase information."""
    print("""
PHASE 2 WORKFLOW:
─────────────────────────────────────────────────────────────────────────

Step 1: Data Preparation
  └─ Split cleaned corpus into:
     • Validation set (20%): Both annotators label same sentences
     • Training set A (40%): Unique to Annotator A
     • Training set B (40%): Unique to Annotator B

Step 2: Annotation
  └─ Each annotator assigns labels:
     • 0 = NEGATIVE (disapproval, criticism, sadness)
     • 1 = POSITIVE (approval, praise, happiness)
     • 2 = NEUTRAL (factual, informational, objective)
  
  └─ Follow 3 core linguistic rules:
     1. Sarcasm Law: Judge intent, not surface words
     2. Final-Clause Rule: Use terminal clause for compound sentences
     3. Objective Boundary: Facts without evaluation = Neutral

Step 3: Inter-Annotator Agreement (Cohen's Kappa)
  └─ Calculate κ on validation set
  └─ κ ≥ 0.60 → ACCEPT (publication-ready)
  └─ κ < 0.60 → Run consensus mediation

Step 4: Consensus Mediation (if needed)
  └─ Identify disagreement cases
  └─ Discuss & resolve
  └─ Re-label & recalculate κ

Step 5: Finalization
  └─ Merge all annotations
  └─ Generate statistics
  └─ Output: 04_annotated_dataset.jsonl (ready for Phase 3)

""")


def step1_data_split():
    """Step 1: Split dataset."""
    print("\n" + "="*80)
    print("STEP 1: DATA SPLITTING")
    print("="*80 + "\n")
    
    try:
        from phase2_data_split import main as split_main
        split_main()
        return True
    except Exception as e:
        print(f"✗ Error in data splitting: {e}")
        return False


def step2_annotation(auto_mode=False):
    """Step 2: Complete annotations (human or simulated)."""
    print("\n" + "="*80)
    print("STEP 2: ANNOTATION")
    print("="*80 + "\n")
    
    if auto_mode:
        print("Running annotation simulation for demonstration (auto mode)...\n")
        choice = "1"
    else:
        print("Choose annotation mode:")
        print("  [1] Simulate annotations (for demo)")
        print("  [2] Manual annotation (provide files)")
        print()
        
        choice = input("Enter choice [1-2] (default: 1): ").strip() or "1"
    
    if choice == "1":
        print("Running annotation simulation for demonstration...\n")
        try:
            from phase2_annotation_simulator import main as sim_main
            sim_main()
            return True
        except Exception as e:
            print(f"✗ Error in simulation: {e}")
            return False
    else:
        print("\nManual annotation mode:")
        print("  1. Open these files and add annotations:")
        print("     - data/03_validation_partition.jsonl")
        print("     - data/03_training_annotator_A.jsonl")
        print("     - data/03_training_annotator_B.jsonl")
        print()
        print("  2. For each sentence, set:")
        print("     - 'label': 0 (Negative) | 1 (Positive) | 2 (Neutral)")
        print("     - 'confidence': HIGH | MEDIUM | LOW")
        print("     - 'notes': Optional reasoning for uncertain cases")
        print()
        print("  3. Run Step 3 when complete")
        if not auto_mode:
            input("Press Enter when annotations are complete...")
        return True


def step3_kappa_calculation():
    """Step 3: Calculate Cohen's Kappa."""
    print("\n" + "="*80)
    print("STEP 3: INTER-ANNOTATOR AGREEMENT (Cohen's Kappa)")
    print("="*80 + "\n")
    
    try:
        from phase2_kappa_calculation import main as kappa_main
        kappa_main()
        return True
    except Exception as e:
        print(f"✗ Error in kappa calculation: {e}")
        return False


def step4_consensus_mediation(auto_mode=False):
    """Step 4: Consensus mediation (if needed)."""
    print("\n" + "="*80)
    print("STEP 4: CONSENSUS MEDIATION (if κ < 0.60)")
    print("="*80 + "\n")
    
    print("⚠ Consensus Mediation Protocol:")
    print()
    print("If Cohen's Kappa < 0.60 (moderate agreement):")
    print("  1. Review disagreement cases from kappa report")
    print("  2. Discuss each case:")
    print("     - Which linguistic rule applies?")
    print("     - Is it sarcasm, compound, or objective?")
    print("     - Reach consensus label")
    print("  3. Update annotations in the files")
    print("  4. Re-run Step 3 to recalculate κ")
    print()
    print("If κ ≥ 0.60: Proceed to Step 5 directly")
    print()
    
    if not auto_mode:
        input("Press Enter to continue...")
    return True


def step5_finalization():
    """Step 5: Merge and finalize."""
    print("\n" + "="*80)
    print("STEP 5: FINALIZATION - MERGE ANNOTATIONS")
    print("="*80 + "\n")
    
    try:
        from phase2_merge_annotations import main as merge_main
        merge_main()
        return True
    except Exception as e:
        print(f"✗ Error in merging: {e}")
        return False


def run_full_workflow():
    """Run complete Phase 2 workflow."""
    print("\n" + "="*80)
    print("PHASE 2 - COMPLETE WORKFLOW")
    print("="*80)
    
    steps = [
        ("Data Splitting", lambda: step1_data_split()),
        ("Annotation", lambda: step2_annotation(auto_mode=True)),
        ("Cohen's Kappa Calculation", lambda: step3_kappa_calculation()),
        ("Consensus Mediation", lambda: step4_consensus_mediation(auto_mode=True)),
        ("Finalization", lambda: step5_finalization()),
    ]
    
    for i, (step_name, step_func) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {step_name}")
        
        if not step_func():
            print(f"\n✗ Workflow halted at step {i}")
            return False
    
    print("\n" + "="*80)
    print("✓ PHASE 2 COMPLETE!")
    print("="*80)
    print()
    print("Results:")
    print("  • 04_annotated_dataset.jsonl → Final labeled dataset")
    print("  • 04_annotation_statistics.json → Dataset statistics")
    print("  • 07_kappa_report.json → Inter-annotator agreement report")
    print()
    print("Ready for Phase 3: Model Training & Benchmarking")
    print()
    
    return True


def run_step_by_step():
    """Run individual steps."""
    print("""
Select individual steps:
  [1] Step 1 - Data Splitting
  [2] Step 2 - Annotation (Simulate or Manual)
  [3] Step 3 - Calculate Cohen's Kappa
  [4] Step 4 - Consensus Mediation
  [5] Step 5 - Merge Annotations
  [0] Exit
""")
    
    steps = {
        "1": ("Data Splitting", step1_data_split),
        "2": ("Annotation", step2_annotation),
        "3": ("Cohen's Kappa", step3_kappa_calculation),
        "4": ("Consensus Mediation", step4_consensus_mediation),
        "5": ("Finalization", step5_finalization),
    }
    
    while True:
        choice = input("\nEnter choice [0-5]: ").strip()
        
        if choice == "0":
            break
        elif choice in steps:
            step_name, step_func = steps[choice]
            print(f"\nRunning: {step_name}\n")
            step_func()
        else:
            print("Invalid choice")


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Nepali Sentiment Analysis - Phase 2 Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete workflow
  python phase2_orchestrator.py --full

  # Run step by step
  python phase2_orchestrator.py --step-by-step

  # Run specific step
  python phase2_orchestrator.py --step 1
        """
    )
    
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run complete Phase 2 workflow"
    )
    
    parser.add_argument(
        "--step-by-step",
        action="store_true",
        help="Run steps interactively"
    )
    
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run specific step (1-5)"
    )
    
    args = parser.parse_args()
    
    show_banner()
    
    if args.full:
        run_full_workflow()
    elif args.step:
        step_map = {
            1: ("Data Splitting", step1_data_split),
            2: ("Annotation", step2_annotation),
            3: ("Cohen's Kappa", step3_kappa_calculation),
            4: ("Consensus Mediation", step4_consensus_mediation),
            5: ("Finalization", step5_finalization),
        }
        step_name, step_func = step_map[args.step]
        print(f"\nRunning: {step_name}\n")
        step_func()
    else:
        # Interactive menu
        print("""
Choose mode:
  [1] Run complete Phase 2 workflow
  [2] Run step-by-step (interactive)
  [0] Exit
""")
        
        mode = input("Enter choice [0-2]: ").strip()
        
        if mode == "1":
            run_full_workflow()
        elif mode == "2":
            run_step_by_step()


if __name__ == "__main__":
    main()
