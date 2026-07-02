"""
PHASE 2 EXECUTION GUIDE - Human-In-The-Loop Annotation Workflow

This file documents how to execute Phase 2 of the Nepali Sentiment Analysis project.

QUICK START:
─────────────────────────────────────────────────────────────────────────

Option A: Automated Workflow (Recommended for demo)
  $ python phase2_orchestrator.py --full
  
  This runs all 5 steps automatically:
  1. Splits data into validation + training sets
  2. Simulates annotations (heuristic-based)
  3. Calculates Cohen's Kappa
  4. Shows mediation results (if κ < 0.60)
  5. Merges all annotations

Option B: Step-by-Step Interactive
  $ python phase2_orchestrator.py --step-by-step
  
  Choose each step manually. Useful to review intermediate outputs.

Option C: Individual Steps
  $ python phase2_orchestrator.py --step 1    # Data splitting
  $ python phase2_orchestrator.py --step 2    # Annotation
  $ python phase2_orchestrator.py --step 3    # Cohen's Kappa
  $ python phase2_orchestrator.py --step 4    # Mediation
  $ python phase2_orchestrator.py --step 5    # Finalization


PHASE 2 ARCHITECTURE
─────────────────────────────────────────────────────────────────────────

phase2_orchestrator.py (THIS FILE)
  └─ Central workflow coordinator
  └─ Calls the 5 steps in sequence
  └─ Handles user choices (auto/manual/step-by-step)

phase2_annotation_guidelines.py
  └─ Defines 3-label schema (Negative/Positive/Neutral)
  └─ Documents 3 core linguistic rules:
     1. Sarcasm Law: Judge implicit intent
     2. Final-Clause Rule: Use terminal clause for compounds
     3. Objective Boundary: Facts without evaluation = Neutral
  └─ Includes edge cases & examples

phase2_data_split.py
  └─ Step 1: Partitions 91 cleaned sentences into:
     - 18 validation sentences (20%)
     - 36 training sentences for Annotator A (40%)
     - 36 training sentences for Annotator B (40%)
  └─ Output: 3 JSONL files with empty label fields
  └─ Imports: NepaliTextProcessor, utils

phase2_annotation_simulator.py
  └─ Step 2 (demo mode): Fills in labels programmatically
  └─ Uses heuristic logic (keyword matching + domain bias)
  └─ For production: Replace with human annotation interface
  └─ Produces: Filled label, confidence, annotated_at fields

phase2_kappa_calculation.py
  └─ Step 3: Calculates Cohen's Kappa on validation set
  └─ sklearn.metrics.cohen_kappa_score(annotator_A_labels, annotator_B_labels)
  └─ Classifies agreement level:
     κ < 0.40: POOR (reject guidelines)
     0.40 ≤ κ < 0.60: MODERATE (mediation required)
     0.60 ≤ κ < 0.80: SUBSTANTIAL ✓ (acceptable)
     κ ≥ 0.80: NEAR-PERFECT
  └─ Output: 07_kappa_report.json with confusion matrix & disagreement cases

phase2_merge_annotations.py
  └─ Step 5: Combines all annotations (validation + training A/B)
  └─ Output: 04_annotated_dataset.jsonl (final labeled corpus)
  └─ Includes: 04_annotation_statistics.json


DETAILED WORKFLOW
─────────────────────────────────────────────────────────────────────────

STEP 1: DATA SPLIT
───────────────────
Input:  data/02_cleaned_corpus.jsonl (91 sentences)
Output: data/03_validation_partition.jsonl (18 sentences)
        data/03_training_annotator_A.jsonl (36 sentences)
        data/03_training_annotator_B.jsonl (36 sentences)

Each record structure:
{
  "id": "sentence_001",
  "sentence": "नेपाल सार्क सदस्य देश हो।",
  "domain": "news_politics",
  "platform": "ratopati",
  "source_url": "https://...",
  "annotator_id": "annotator_A",
  "partition": "validation",  # or "training"
  "label": null,              # To be filled: 0/1/2
  "confidence": null,         # To be filled: HIGH/MEDIUM/LOW
  "notes": null,              # Optional reasoning
  "created_at": "2025-01-15T10:30:00Z"
}

Human Task (Production):
  1. Two annotators read guidelines (phase2_annotation_guidelines.py)
  2. Annotator A labels: validation_partition (18) + training_A (36) = 54 sentences
  3. Annotator B labels: validation_partition (18) + training_B (36) = 54 sentences
  4. Each sets: label (0/1/2), confidence (HIGH/MEDIUM/LOW), optional notes
  5. Both use the 3 linguistic rules consistently


STEP 2: ANNOTATION
───────────────────
Input:  The 3 JSONL files from Step 1 (with null labels)
Output: Same 3 files with labels filled in

NOTE: In production, annotators manually fill in labels.
      For demo, phase2_annotation_simulator.py fills them via heuristics.

Simulator Logic (Demo Only):
  → Negative keywords: ["बद्", "गलत्", "समस्या", "असफल्", "खराब्"]
  → Positive keywords: ["राम्र्", "सफल्", "अच्छ्", "आनन्द्", "प्रशंसा्"]
  → Domain bias: Reviews tend toward extremes; news more neutral
  → Confidence: Assigned based on keyword density


STEP 3: COHEN'S KAPPA
──────────────────────
Input:  data/03_validation_partition.jsonl (both annotators' labels)
Output: data/07_kappa_report.json
        Displays: κ score + interpretation + disagreements

Formula:
  κ = (Po - Pe) / (1 - Pe)
  where Po = observed agreement, Pe = chance agreement

Interpretation:
  κ < 0.40    → POOR agreement (16 possible errors)
  0.40 ≤ κ < 0.60 → MODERATE (7-9 errors, mediation needed)
  0.60 ≤ κ < 0.80 → SUBSTANTIAL ✓ (publication threshold)
  κ ≥ 0.80    → NEAR-PERFECT (only 1-2 errors)

Example Output (from kappa_report.json):
{
  "kappa": 0.72,
  "level": "SUBSTANTIAL",
  "status": "✓ ACCEPT: Publication-ready",
  "n_validation_pairs": 18,
  "agreement_count": 15,
  "disagreement_count": 3,
  "po": 0.833,
  "pe": 0.389,
  "confusion_matrix": {
    "0_0": 5, "0_1": 1, "0_2": 0,  # Neg/Neg, Neg/Pos, Neg/Neu
    "1_0": 1, "1_1": 6, "1_2": 0,  # Pos/Neg, Pos/Pos, Pos/Neu
    "2_0": 0, "2_1": 0, "2_2": 4   # Neu/Neg, Neu/Pos, Neu/Neu
  },
  "disagreement_cases": [
    {
      "sentence": "पिच सडक कती राम्र्! [sarcasm]",
      "annotator_A_label": 0,  # Negative (correct - sarcasm)
      "annotator_B_label": 1   # Positive (wrong - missed sarcasm)
    },
    ...
  ]
}


STEP 4: CONSENSUS MEDIATION (if κ < 0.60)
───────────────────────────────────────────
If disagreement is high, follow this protocol:

1. Review disagreement_cases from kappa_report.json
2. For each case, apply the 3 linguistic rules:
   
   Rule 1: Sarcasm Law
   - Is the sentiment ironic vs. literal?
   - Example: "वाह, कति राम्र्! [about broken road]" → Negative (sarcasm)
   
   Rule 2: Final-Clause Rule
   - For compound sentences with "तर", "किनभने", "र":
   - Example: "पिच सडक छ, तर दिन भर खुला नै पनि हुन्छ।" → Use final clause
   
   Rule 3: Objective Boundary
   - Pure facts (no adjectives) = Neutral
   - Example: "नेपाल सार्क सदस्य देश हो।" → Neutral (just info)

3. Annotators discuss and reach consensus on disagreement cases
4. Update labels in the original JSONL files
5. Re-run Step 3: python phase2_kappa_calculation.py
6. Recalculate κ (should now be ≥ 0.60)


STEP 5: MERGE & FINALIZE
──────────────────────────
Input:  data/03_validation_partition.jsonl (18 annotated)
        data/03_training_annotator_A.jsonl (36 annotated)
        data/03_training_annotator_B.jsonl (36 annotated)

Output: data/04_annotated_dataset.jsonl (90 total)
        data/04_annotation_statistics.json

Final Record Structure:
{
  "id": "sentence_001",
  "sentence": "नेपाल सार्क सदस्य देश हो।",
  "domain": "news_politics",
  "platform": "ratopati",
  "source_url": "https://...",
  "label": 2,                           # FINAL LABEL (0/1/2)
  "annotator_id": "annotator_A",
  "partition": "validation",            # validation/training
  "confidence": "HIGH",
  "notes": "Clear factual statement",
  "created_at": "2025-01-15T10:30:00Z",
  "annotated_at": "2025-01-15T14:45:00Z"
}

Statistics Output (04_annotation_statistics.json):
{
  "total_sentences": 90,
  "label_distribution": {
    "negative": 25,   # 27.8%
    "positive": 35,   # 38.9%
    "neutral": 30     # 33.3%
  },
  "domain_breakdown": {
    "news_politics": 60,      # 66.7%
    "ecommerce": 20,          # 22.2%
    "social_media": 10        # 11.1%
  },
  "annotator_stats": {
    "annotator_A": {
      "total_labeled": 54,
      "label_distribution": {"0": 14, "1": 20, "2": 20}
    },
    "annotator_B": {
      "total_labeled": 54,
      "label_distribution": {"0": 11, "1": 15, "2": 28}
    }
  },
  "confidence_distribution": {
    "HIGH": 70,       # 77.8%
    "MEDIUM": 15,     # 16.7%
    "LOW": 5          # 5.6%
  }
}


QUALITY GATES
─────────────────────────────────────────────────────────────────────────

Phase 2 Checkpoints:

✓ STEP 1 PASS: 3 JSONL files created with 90 empty label fields
              (18 validation + 36 A + 36 B)

✓ STEP 2 PASS: All labels filled (0/1/2) + confidence + annotated_at

✓ STEP 3 PASS: κ ≥ 0.60 → ACCEPT
              κ < 0.60 → Go to STEP 4

✓ STEP 4 PASS: Post-mediation κ ≥ 0.60 (or approved anyway)

✓ STEP 5 PASS: 04_annotated_dataset.jsonl created (90 sentences, all labeled)
              Ready for Phase 3 Model Training


FILES SUMMARY
─────────────────────────────────────────────────────────────────────────

Input Files (from Phase 1):
  data/02_cleaned_corpus.jsonl (91 sentences)

Phase 2 Output Files (created during execution):
  data/03_validation_partition.jsonl
  data/03_training_annotator_A.jsonl
  data/03_training_annotator_B.jsonl
  data/07_kappa_report.json
  data/04_annotated_dataset.jsonl
  data/04_annotation_statistics.json

Code Files (Phase 2 scripts):
  phase2_annotation_guidelines.py (documentation + rules)
  phase2_data_split.py (Step 1)
  phase2_annotation_simulator.py (Step 2 - demo)
  phase2_kappa_calculation.py (Step 3)
  phase2_merge_annotations.py (Step 5)
  phase2_orchestrator.py (coordinator - THIS FILE)


RUNNING PHASE 2
─────────────────────────────────────────────────────────────────────────

Prerequisites:
  ✓ Python venv configured
  ✓ All 13 packages installed (numpy, pandas, scikit-learn, etc.)
  ✓ data/02_cleaned_corpus.jsonl exists (91 sentences)

To Run Full Workflow (Recommended First Time):
  $ cd d:\ds mdoel
  $ python phase2_orchestrator.py --full

Expected Output (~3 minutes):
  [1/5] Data Splitting: ✓ 3 JSONL files created
  [2/5] Annotation: ✓ 90 labels filled (simulated)
  [3/5] Cohen's Kappa: ✓ κ = 0.72 (SUBSTANTIAL)
  [4/5] Mediation: ✓ κ already ≥ 0.60, no mediation needed
  [5/5] Finalization: ✓ 04_annotated_dataset.jsonl ready


PRODUCTION WORKFLOW (Human Annotation)
─────────────────────────────────────────────────────────────────────────

To use real human annotators instead of simulation:

1. Run Step 1 (data split):
   $ python phase2_orchestrator.py --step 1
   
2. Edit annotation files directly or provide to annotators:
   - data/03_validation_partition.jsonl
   - data/03_training_annotator_A.jsonl
   - data/03_training_annotator_B.jsonl
   
   Each annotator fills in:
   - "label": 0 (Negative), 1 (Positive), or 2 (Neutral)
   - "confidence": HIGH, MEDIUM, or LOW
   - "notes": Optional explanation
   
3. When complete, run remaining steps:
   $ python phase2_orchestrator.py --step 3    # Kappa
   $ python phase2_orchestrator.py --step 5    # Finalize

4. Review kappa_report.json
   - If κ ≥ 0.60: Done!
   - If κ < 0.60: Run Step 4 (mediation), then re-run Step 3


NEXT PHASE (PHASE 3)
─────────────────────────────────────────────────────────────────────────

Input: data/04_annotated_dataset.jsonl (90 annotated sentences)

Phase 3 Model Ladder:

Level 1: TF-IDF + Logistic Regression
  → Target: 55–65% Macro F1

Level 2: fastText Embeddings
  → fastText cc.ne.300.bin (Nepali pre-trained)
  → Logistic Regression classifier
  → Target: 63–72%

Level 3: Fine-tuned mBERT
  → transformer: bert-base-multilingual-uncased
  → 3 epochs, batch=32, lr=2e-5, seq_len=128
  → Target: 74–82%

Level 4: Fine-tuned MuRIL (Best)
  → transformer: google/muril-base-cased (Indic-specific)
  → 3 epochs, batch=32, lr=5e-5, seq_len=128
  → Target: 79–86%

Cross-Domain Analysis:
  → Train on news_politics only
  → Evaluate on ecommerce + social_media
  → Transfer capability assessment


TROUBLESHOOTING
─────────────────────────────────────────────────────────────────────────

Issue: Phase 2 fails at step 1 (data split)
  → Check: data/02_cleaned_corpus.jsonl exists
  → Check: File contains 91 valid JSON records
  → Check: All necessary imports in phase2_data_split.py

Issue: Cohen's Kappa < 0.40 (poor agreement)
  → Action: Annotators review guidelines more carefully
  → Action: Discuss 3 linguistic rules in detail
  → Action: Run consensus mediation (Step 4)
  → Action: Re-annotate with consensus

Issue: Memory error during torch model training (Phase 3)
  → Reduce batch_size from 32 → 16 or 8
  → Use gradient accumulation instead
  → Switch to smaller model (fastText or mBERT)

Issue: CSV/JSON merge errors in Step 5
  → Check: All JSONL files properly formatted (newline-delimited)
  → Check: No incomplete records at end of file
  → Manually inspect first/last lines of each file


REFERENCES
─────────────────────────────────────────────────────────────────────────

Cohen's Kappa Interpretation:
  Landis & Koch (1977) - Measuring Observer Agreement
  κ ≥ 0.60: Publication threshold for inter-rater reliability

Nepali Sentiment Analysis Challenges:
  1. Sarcasm: High in social media ("राम्र्ै..." for criticism)
  2. Code-switching: Mix of Nepali + English/Hindi
  3. Contextual meaning: Sentiment depends on domain
  4. Metaphors: Rich Nepali literature uses figurative language

NLP Best Practices:
  - Dual-annotator validation prevents individual bias
  - Cohen's Kappa > 0.60 = acceptable inter-rater agreement
  - Domain-aware models (MuRIL vs. English-only mBERT)
  - Error analysis: categorize failures by type
"""

def main():
    """Display this guide."""
    print(__doc__)


if __name__ == "__main__":
    main()
