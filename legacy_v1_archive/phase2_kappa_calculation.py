"""
Phase 2: Cohen's Kappa calculation and inter-annotator agreement analysis.

Cohen's Kappa (κ) measures agreement between two raters beyond chance.
Formula: κ = (P_o - P_e) / (1 - P_e)
  P_o = observed agreement
  P_e = expected agreement by chance

Interpretation:
  κ < 0.40   → Poor (reject guidelines)
  0.40 ≤ κ < 0.60 → Moderate (needs mediation)
  0.60 ≤ κ < 0.80 → Substantial ✓ (acceptable)
  κ ≥ 0.80   → Near-perfect (excellent)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

try:
    from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
except ImportError:
    print("✗ scikit-learn not found. Install: pip install scikit-learn")
    sys.exit(1)


class KappaCalculator:
    """Calculate Cohen's Kappa for validation set annotations."""
    
    def __init__(self, validation_file: Path):
        """
        Initialize calculator.
        
        Args:
            validation_file: Path to 03_validation_partition.jsonl
        """
        self.validation_file = validation_file
        self.annotations_A = {}
        self.annotations_B = {}
        self.load_annotations()
    
    def load_annotations(self):
        """Load annotations from both annotators."""
        if not self.validation_file.exists():
            raise FileNotFoundError(f"Validation file not found: {self.validation_file}")
        
        with open(self.validation_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    record = json.loads(line)
                    
                    # Extract base sentence ID (remove _A or _B suffix)
                    record_id = record['id'].replace('_A', '').replace('_B', '')
                    
                    if record['annotator_id'] == 'annotator_A':
                        self.annotations_A[record_id] = record
                    else:
                        self.annotations_B[record_id] = record
                
                except json.JSONDecodeError:
                    continue
        
        print(f"[*] Loaded annotations:")
        print(f"    Annotator A: {len(self.annotations_A)} sentences")
        print(f"    Annotator B: {len(self.annotations_B)} sentences")
    
    def align_annotations(self) -> Tuple[List[int], List[int], List[str]]:
        """
        Align annotations from both annotators for comparison.
        
        Returns:
            (labels_A, labels_B, common_ids)
        """
        common_ids = sorted(set(self.annotations_A.keys()) & set(self.annotations_B.keys()))
        
        labels_A = [self.annotations_A[idx]['label'] for idx in common_ids]
        labels_B = [self.annotations_B[idx]['label'] for idx in common_ids]
        
        # Filter out None labels (unannotated)
        valid_indices = [i for i, (la, lb) in enumerate(zip(labels_A, labels_B)) 
                        if la is not None and lb is not None]
        
        labels_A = [labels_A[i] for i in valid_indices]
        labels_B = [labels_B[i] for i in valid_indices]
        common_ids = [common_ids[i] for i in valid_indices]
        
        print(f"\n[*] Aligned annotations:")
        print(f"    Common sentences: {len(common_ids)}")
        print(f"    Annotated pairs: {len([i for i in valid_indices])}")
        
        return labels_A, labels_B, common_ids
    
    def calculate_kappa(self, labels_A: List[int], labels_B: List[int]) -> float:
        """
        Calculate Cohen's Kappa.
        
        Args:
            labels_A: Labels from Annotator A
            labels_B: Labels from Annotator B
            
        Returns:
            Cohen's Kappa coefficient (0.0 to 1.0)
        """
        if len(labels_A) != len(labels_B):
            raise ValueError("Label lists must have equal length")
        
        if len(labels_A) == 0:
            raise ValueError("No annotated data to compare")
        
        kappa = cohen_kappa_score(labels_A, labels_B)
        return kappa
    
    def classify_agreement(self, kappa: float) -> Tuple[str, str]:
        """Classify kappa score and provide interpretation."""
        if kappa < 0.40:
            level = "POOR"
            action = "❌ REJECT: Reconstruct guidelines and restart validation"
        elif kappa < 0.60:
            level = "MODERATE"
            action = "⚠ MEDIATION: Run consensus rounds and re-label disagreements"
        elif kappa < 0.80:
            level = "SUBSTANTIAL"
            action = "✓ ACCEPT: Publication-ready agreement level"
        else:
            level = "NEAR-PERFECT"
            action = "✓✓ EXCELLENT: Outstanding inter-annotator agreement"
        
        return level, action
    
    def get_confusion_matrix(self, labels_A: List[int], labels_B: List[int]) -> Dict:
        """
        Generate confusion matrix.
        
        Shows where annotators agree/disagree.
        """
        cm = confusion_matrix(labels_A, labels_B, labels=[0, 1, 2])
        
        label_names = {0: "Negative", 1: "Positive", 2: "Neutral"}
        
        result = {
            "confusion_matrix": cm.tolist(),
            "annotations": {
                "rows": "Annotator A (true label)",
                "cols": "Annotator B (predicted label)"
            }
        }
        
        return result
    
    def find_disagreements(self, labels_A: List[int], labels_B: List[int], 
                           common_ids: List[str]) -> List[Dict]:
        """Find sentences where annotators disagree."""
        disagreements = []
        
        for idx, (la, lb, sid) in enumerate(zip(labels_A, labels_B, common_ids)):
            if la != lb:
                disagreements.append({
                    "sentence_id": sid,
                    "sentence": self.annotations_A[sid]['sentence'],
                    "domain": self.annotations_A[sid]['domain'],
                    "annotator_A_label": la,
                    "annotator_B_label": lb,
                    "notes_A": self.annotations_A[sid].get('notes', ''),
                    "notes_B": self.annotations_B[sid].get('notes', '')
                })
        
        return disagreements


def main():
    """Calculate and report Cohen's Kappa."""
    
    print("=" * 80)
    print(" PHASE 2: INTER-ANNOTATOR AGREEMENT ANALYSIS (Cohen's Kappa)")
    print("=" * 80)
    print()
    
    validation_file = Path("d:/ds mdoel/data/03_validation_partition.jsonl")
    
    if not validation_file.exists():
        print(f"✗ Validation file not found: {validation_file}")
        print("  Please run: python phase2_data_split.py")
        sys.exit(1)
    
    try:
        calculator = KappaCalculator(validation_file)
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    
    # Align and extract labels
    labels_A, labels_B, common_ids = calculator.align_annotations()
    
    if len(labels_A) == 0:
        print("\n✗ No annotated pairs found in validation set.")
        print("  Please complete annotations first.")
        sys.exit(1)
    
    # Calculate Cohen's Kappa
    kappa = calculator.calculate_kappa(labels_A, labels_B)
    level, action = calculator.classify_agreement(kappa)
    
    print()
    print("=" * 80)
    print(" COHEN'S KAPPA RESULT")
    print("=" * 80)
    print()
    print(f"  κ = {kappa:.4f}")
    print(f"  Agreement Level: {level}")
    print(f"  Action: {action}")
    print()
    
    # Agreement statistics
    agreement_count = sum(1 for a, b in zip(labels_A, labels_B) if a == b)
    disagreement_count = len(labels_A) - agreement_count
    agreement_pct = (agreement_count / len(labels_A)) * 100
    
    print(f"Agreement Statistics:")
    print(f"  Total sentence pairs: {len(labels_A)}")
    print(f"  Agreed: {agreement_count} ({agreement_pct:.1f}%)")
    print(f"  Disagreed: {disagreement_count} ({100-agreement_pct:.1f}%)")
    print()
    
    # Confusion matrix
    cm_data = calculator.get_confusion_matrix(labels_A, labels_B)
    cm = cm_data['confusion_matrix']
    
    print("Confusion Matrix (Annotator A vs B):")
    print("           Neg    Pos    Neu")
    labels = ["Neg", "Pos", "Neu"]
    for i, label in enumerate(labels):
        print(f"  {label}  {cm[i][0]:3d}  {cm[i][1]:3d}  {cm[i][2]:3d}")
    print()
    
    # Label distribution
    print("Label Distribution:")
    label_names = {0: "Negative", 1: "Positive", 2: "Neutral"}
    
    label_dist_A = {0: 0, 1: 0, 2: 0}
    label_dist_B = {0: 0, 1: 0, 2: 0}
    
    for label in labels_A:
        label_dist_A[label] += 1
    for label in labels_B:
        label_dist_B[label] += 1
    
    print(f"  Annotator A: Neg={label_dist_A[0]} Pos={label_dist_A[1]} Neu={label_dist_A[2]}")
    print(f"  Annotator B: Neg={label_dist_B[0]} Pos={label_dist_B[1]} Neu={label_dist_B[2]}")
    print()
    
    # Find disagreements
    disagreements = calculator.find_disagreements(labels_A, labels_B, common_ids)
    
    if disagreements:
        print("=" * 80)
        print(f" DISAGREEMENT CASES ({len(disagreements)} total)")
        print("=" * 80)
        print()
        
        for i, case in enumerate(disagreements[:5], 1):  # Show first 5
            print(f"{i}. {case['sentence'][:60]}...")
            print(f"   Domain: {case['domain']}")
            print(f"   A → {label_names[case['annotator_A_label']]} | B → {label_names[case['annotator_B_label']]}")
            if case['notes_A']:
                print(f"   A notes: {case['notes_A']}")
            if case['notes_B']:
                print(f"   B notes: {case['notes_B']}")
            print()
        
        if len(disagreements) > 5:
            print(f"   ... and {len(disagreements) - 5} more disagreement cases")
            print()
    
    # Save report
    report = {
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "cohen_kappa": float(kappa),
        "agreement_level": level,
        "total_pairs": len(labels_A),
        "agreed": agreement_count,
        "disagreed": disagreement_count,
        "agreement_percentage": agreement_pct,
        "confusion_matrix": cm,
        "label_distribution": {
            "annotator_A": label_dist_A,
            "annotator_B": label_dist_B
        },
        "disagreement_cases": disagreements[:10],  # Save first 10 for review
        "action_required": action,
        "interpretation": {
            "<0.40": "Poor - Reject & reconstruct guidelines",
            "0.40-0.60": "Moderate - Run consensus mediation",
            "0.60-0.80": "Substantial - Accept ✓",
            "≥0.80": "Near-perfect - Excellent"
        }
    }
    
    report_file = Path("d:/ds mdoel/data/07_kappa_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Report saved: {report_file}")
    print()
    
    # Guidance for next steps
    print("=" * 80)
    print(" NEXT STEPS")
    print("=" * 80)
    print()
    
    if kappa < 0.60:
        print("⚠ Consensus Mediation Required:")
        print("  1. Review the disagreement cases above")
        print("  2. Discuss interpretation of annotation guidelines")
        print("  3. Re-label together or apply tie-breaker")
        print("  4. Re-run kappa calculation")
        print()
    else:
        print("✓ Quality Gate Passed! Proceed with:")
        print("  1. Annotator A labels remaining training set")
        print("  2. Annotator B labels remaining training set")
        print("  3. Run: python phase2_merge_annotations.py")
        print("  4. Generate final: 04_annotated_dataset.jsonl")
        print()
    
    print("=" * 80)


if __name__ == "__main__":
    main()
