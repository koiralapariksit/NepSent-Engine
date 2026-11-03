"""
NEPALI SENTIMENT ANALYSIS - ANNOTATION GUIDELINES
Phase 2: Human-In-The-Loop Data Labeling

Target Label Schema (3 Classes):
  0 = Negative
  1 = Positive
  2 = Neutral

Mathematical Quality Gate:
  Cohen's Kappa (κ) ≥ 0.60 for publication-ready agreement
"""

ANNOTATION_GUIDELINES = """
================================================================================
                    NEPALI SENTIMENT ANNOTATION GUIDELINES
                            Version 1.0 (Phase 2)
================================================================================

TASK OVERVIEW
─────────────────────────────────────────────────────────────────────────────
You will assign each Nepali sentence to ONE of three sentiment classes:

  0 = NEGATIVE    → Expresses disapproval, criticism, sadness, anger, distress
  1 = POSITIVE    → Expresses approval, praise, happiness, satisfaction, support
  2 = NEUTRAL     → Factual, informational, opinion-neutral, objective rendering


LABEL DEFINITIONS (WITH EXAMPLES)
─────────────────────────────────────────────────────────────────────────────

[NEGATIVE = 0]
Definition: The sentence explicitly conveys a negative sentiment, criticism,
complaint, disapproval, or distressing emotion.

Examples:
  • "यो उत्पाद धेरै खराब छ र मैले गलत निर्णय गरें।"
    (This product is very bad and I made a wrong decision.)
    → NEGATIVE

  • "नेपालको अर्थनीति गम्भीर संकटमा परेको छ।"
    (Nepal's economy has fallen into a severe crisis.)
    → NEGATIVE

  • "मलाई यो काम गर्न बिल्कुल मनपर्दैन।"
    (I absolutely don't like doing this work.)
    → NEGATIVE


[POSITIVE = 1]
Definition: The sentence explicitly conveys positive sentiment, praise,
approval, happiness, satisfaction, or favorable judgment.

Examples:
  • "यो उत्पाद धेरै राम्रो छ र मलाई मनपर्यो।"
    (This product is very good and I liked it.)
    → POSITIVE

  • "नेपालको विकास गति अचम्मित छ।"
    (Nepal's development pace is impressive.)
    → POSITIVE

  • "मेरो नयाँ काम सँग मैं भरपूर खुस छु।"
    (I am fully satisfied with my new job.)
    → POSITIVE


[NEUTRAL = 2]
Definition: The sentence is FACTUAL, INFORMATIONAL, OBJECTIVE, or expresses
no clear sentiment. Includes dry data, timelines, official updates, and
neutral opinions without evaluative weight.

Examples:
  • "काठमाडौंमा गत वर्ष ५ लाख पर्यटक आएका थिए।"
    (5 lakh tourists came to Kathmandu last year.)
    → NEUTRAL (factual data)

  • "सरकारले नयाँ शिक्षा नीति घोषणा गरेको छ।"
    (The government announced a new education policy.)
    → NEUTRAL (official announcement, no evaluation)

  • "नेपाल दक्षिण एशिया क्षेत्रमा अवस्थित छ।"
    (Nepal is located in the South Asian region.)
    → NEUTRAL (geographical fact)


================================================================================
                           SPECIAL EDGE CASES
================================================================================

[RULE 1: THE SARCASM LAW]
─────────────────────────────────────────────────────────────────────────────
Rule: Disregard literal positive/negative words if the STRUCTURAL INTENT is
ironic or sarcastic. Judge by IMPLIED sentiment, not surface words.

Example:
  Sentence: "वाह, कति राम्रो पिच सडक!" (displayed with image of potholes)
  Literal meaning: "Wow, what a beautiful road!"
  Actual intent: Sarcastic criticism of bad road conditions
  Label: NEGATIVE (despite positive words)

Guideline:
  • Always consider CONTEXT clues (tone, punctuation, images, preceding text)
  • Sarcasm detection: Extreme praise for obviously negative scenarios
  • When uncertain, mark separately in notes (annotators will compare)


[RULE 2: THE FINAL-CLAUSE RULE]
─────────────────────────────────────────────────────────────────────────────
Rule: For compound sentences with MIXED POLARITIES, label based on:
  (a) The TERMINAL/FINAL clause (grammatically last sentiment)
  (b) If final is neutral, use DOMINANT SEMANTIC WEIGHT

Example 1 (Terminal priority):
  Sentence: "मेरो पुरानो नोकरी खराब थियो, तर नयाँ नोकरी सुपरको छ।"
  (My old job was bad, but the new job is excellent.)
  Final clause expresses: POSITIVE (सुपरको = excellent)
  Label: POSITIVE

Example 2 (Semantic weight):
  Sentence: "यो उत्पाद सामान्य छ र मैले सिफारस गर्दिन।"
  (This product is average and I would not recommend it.)
  Final clause: Slightly negative (not recommended)
  Semantic weight: Disapproval dominates
  Label: NEGATIVE


[RULE 3: THE OBJECTIVE BOUNDARY]
─────────────────────────────────────────────────────────────────────────────
Rule: Sentences transmitting FACTUAL TIMELINES, DRY DATA, or OFFICIAL UPDATES
WITHOUT descriptive adjectives are strictly NEUTRAL.

Trigger words (neutral context):
  • गर्यो / गरेको / घोषणा (announced, did)
  • छ / विषय (is / regarding)
  • डेटा / संख्या / आँकडा (data / number / statistics)
  • नीति (policy) — unless explicitly criticized/praised
  • समाचार (news) — unless judgment attached
  • अवस्थित (located)

Example (Objective):
  "नेपाल सार्क सदस्य देश हो।"
  (Nepal is a SAARC member state.)
  → NEUTRAL (factual, no evaluation)

Example (With evaluation):
  "नेपाल गर्विलो सार्क सदस्य देश हो।"
  (Nepal is a proud SAARC member state.)
  → POSITIVE (adjective "गर्विलो" = proud adds sentiment)


================================================================================
                       ANNOTATION WORKFLOW
================================================================================

STEP 1: READ SENTENCE CAREFULLY
─────────────────────────────────────────────────────────────────────────────
• Understand ALL words (use dictionary if needed)
• Note tone markers (!, ?, ।)
• Consider potential sarcasm or irony


STEP 2: APPLY THE THREE RULES
─────────────────────────────────────────────────────────────────────────────
☐ Rule 1: Is this SARCASTIC? (Actual intent ≠ surface meaning)
☐ Rule 2: Is this COMPOUND? (Use final clause or semantic weight)
☐ Rule 3: Is this OBJECTIVE? (Pure facts without evaluation)


STEP 3: ASSIGN LABEL
─────────────────────────────────────────────────────────────────────────────
If sentiment is clearly POSITIVE → Label: 1
If sentiment is clearly NEGATIVE → Label: 0
If factual/informational/no evaluation → Label: 2

Confidence scale (optional note):
  HIGH (100% certain) / MEDIUM (75% certain) / LOW (50% uncertain)


STEP 4: RECORD REASONING (IF UNCERTAIN)
─────────────────────────────────────────────────────────────────────────────
For ambiguous cases, add brief note explaining your reasoning.
This helps with consensus mediation if annotators disagree.

Example note:
  "Sentence uses negative words but context suggests sarcasm (Rule 1)"
  "Mixed polarity: positive first clause, but negative final (Rule 2)"
  "Could be fact or mild criticism—borders NEUTRAL/NEGATIVE"


================================================================================
                      INTER-ANNOTATOR AGREEMENT
================================================================================

VALIDATION SET (20% of corpus): 18 SENTENCES

Both annotators will label SAME 18 sentences independently.
Then: Calculate Cohen's Kappa (κ) to measure agreement.

κ Interpretation:
  κ < 0.40   → Poor agreement       → REJECT; reconstruct guidelines
  0.40–0.60  → Moderate agreement  → Run consensus mediation
  0.60–0.80  → Substantial ✓       → ACCEPT (publication-ready)
  ≥ 0.80     → Near-perfect        → EXCELLENT

CONSENSUS MEDIATION (if 0.40 ≤ κ < 0.60):
  1. Identify disagreement cases
  2. Discuss interpretation of rules
  3. Re-label together or use tie-breaker
  4. Recompute κ
  5. If still <0.60, refine guidelines and restart validation


================================================================================
                           ANNOTATION TIPS
================================================================================

✓ DO:
  • Read each sentence 2–3 times
  • Consider cultural context (Nepali idioms, social norms)
  • Note uncertainty; better than forced guesses
  • Use notes for edge cases; helps mediation
  • Re-read rules before starting to keep criteria consistent

✗ DON'T:
  • Rush through sentences
  • Ignore tone markers (!, ?, |।)
  • Assume neutrality by default
  • Over-weight single words (context matters)
  • Second-guess too much after labeling block

⚠ UNCERTAIN ABOUT A SENTENCE?
  → Label your best judgment
  → Add confidence note
  → Annotator disagreement on this case will flag it for mediation


================================================================================
                          DATASET STRUCTURE
================================================================================

Each annotation record contains:
{
  "id": "unique_sentence_id",
  "sentence": "नेपालको अर्थनीति यस वर्ष बढदारी गरिरहेको छ।",
  "domain": "news_politics",
  "platform": "ratopati"
}

YOUR TASK: Add three fields:
{
  ...,
  "label": 1,  # 0=Negative, 1=Positive, 2=Neutral
  "annotator_id": "annotator_A",  # Your identifier
  "confidence": "HIGH"  # HIGH / MEDIUM / LOW (optional)
}

Validation Set: Both annotators → Identical sentences
Training Set: Assigned separately → Annotator A gets unique subset,
                                    Annotator B gets unique subset


================================================================================
                         QUALITY CHECKLIST
================================================================================

Before submitting annotations:

☐ All sentences have labels (0, 1, or 2)
☐ No blank/missing labels
☐ Annotations follow the three main rules
☐ Note added for uncertain cases
☐ Reviewed rules at least once during annotation
☐ Consistent application across blocks (not drifting)


================================================================================
Questions? Refer to:
  1. Examples in each label definition
  2. The three edge case RULES
  3. Validation mediation procedure
================================================================================
"""


LINGUISTIC_BOUNDARY_RULES = {
    "SARCASM_LAW": {
        "short_name": "Disregard Literal Words for Sarcasm",
        "description": "Judge by IMPLIED intent, not surface words. Extreme positive words used for negative scenarios = NEGATIVE.",
        "example": "वाह, कति राम्रो पिच सडक! (said about potholes) → NEGATIVE"
    },
    "FINAL_CLAUSE_RULE": {
        "short_name": "Mixed Polarity → Terminal Clause Priority",
        "description": "Compound sentences: use final clause sentiment OR dominant semantic weight if final is neutral.",
        "example": "XXX खराब छ, तर YYY सुपरको छ। (old job bad, new job excellent) → POSITIVE"
    },
    "OBJECTIVE_BOUNDARY": {
        "short_name": "Factual Statements Without Evaluation = Neutral",
        "description": "Timelines, dry data, official updates without descriptive adjectives → NEUTRAL.",
        "example": "नेपाल सार्क सदस्य देश हो। (Nepal is SAARC member) → NEUTRAL"
    }
}

LABEL_SCHEMA = {
    0: {
        "name": "NEGATIVE",
        "description": "Disapproval, criticism, sadness, anger, distress",
        "keywords": ["खराब", "नराम्रो", "असन्तुष्ट", "गुन्साउनु", "दु:ख"]
    },
    1: {
        "name": "POSITIVE",
        "description": "Approval, praise, happiness, satisfaction, support",
        "keywords": ["राम्रो", "सुपरको", "खुस", "सन्तुष्ट", "तारिफ"]
    },
    2: {
        "name": "NEUTRAL",
        "description": "Factual, informational, objective, no clear sentiment",
        "keywords": ["छ", "अवस्थित", "घोषणा", "डेटा", "नीति"]
    }
}

if __name__ == "__main__":
    print(ANNOTATION_GUIDELINES)
    print("\n\nLinguistic Rules Summary:")
    for rule_name, rule_def in LINGUISTIC_BOUNDARY_RULES.items():
        print(f"\n{rule_name}:")
        print(f"  {rule_def['short_name']}")
        print(f"  {rule_def['description']}")
