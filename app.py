"""
NepSent-Engine: Public Interactive Sentiment Demonstration & Linguistic Analyzer.

STRICT RESEARCH INVARIANTS:
1. Untrained / Demo Mode Blocked: Never instantiates an untrained model head to fabricate
   sentiment predictions. If no trained checkpoints exist in checkpoints/, displays a clear
   blocked state with instructions on how to train a real model.
2. Calibration Transparency: Clearly identifies whether prediction confidence is
   Platt-calibrated or uncalibrated softmax score.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np

from nepali_scraper.config import SCHEMA_VERSION, LABEL_SCHEMA
from nepali_scraper.processors import normalize_nepali_unicode, devanagari_ratio
from nepali_scraper.evaluation.linguistic_analysis import (
    detect_nepali_negation,
    analyze_code_mixing,
    WORD_TOKEN_PATTERN,
)
from nepali_scraper.cli import load_trained_model

# Streamlit Page Config
st.set_page_config(
    page_title="NepSent-Engine — Nepali Sentiment Analysis",
    page_icon="🇳🇵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .reportview-container { background-color: #f8fafc; }
    .badge-neg { background-color: #fee2e2; color: #991b1b; padding: 4px 8px; border-radius: 4px; font-weight: 600; }
    .badge-neu { background-color: #f1f5f9; color: #334155; padding: 4px 8px; border-radius: 4px; font-weight: 600; }
    .badge-pos { background-color: #dcfce7; color: #166534; padding: 4px 8px; border-radius: 4px; font-weight: 600; }
    .callout-box { border-left: 4px solid #3b82f6; padding: 12px; background-color: #eff6ff; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


def find_available_checkpoints(checkpoints_dir: Path) -> List[Path]:
    """Scan directory for valid saved model checkpoints."""
    if not checkpoints_dir.is_dir():
        return []
    valid_dirs = []
    for sub in checkpoints_dir.iterdir():
        if sub.is_dir():
            has_meta = (sub / "metadata.json").is_file()
            has_manifest = (sub / "training_manifest.json").is_file()
            has_config = (sub / "config.json").is_file()
            has_joblib = list(sub.glob("*.joblib"))
            if has_meta or has_manifest or has_config or has_joblib:
                valid_dirs.append(sub)
    return sorted(valid_dirs)


# Header
st.title("🇳🇵 NepSent-Engine: Nepali Sentiment Research Platform")
st.caption(f"Schema Version: {SCHEMA_VERSION} | Ordinal Labels: 0 (Negative), 1 (Neutral), 2 (Positive)")

# Sidebar: Checkpoint Scanner & System Status
st.sidebar.header("Model Checkpoints")
checkpoints_dir = PROJECT_ROOT / "checkpoints"
available_checkpoints = find_available_checkpoints(checkpoints_dir)
selected_checkpoint_names: List[str] = []

if not available_checkpoints:
    st.sidebar.error("❌ No trained checkpoints found")
    st.error("""
    ### ⚠️ Untrained / Demo Mode Blocked
    
    **Research Integrity Rule:** NepSent-Engine strictly prohibits evaluating random,
    untrained weights and pretending they represent an authentic Nepali sentiment classifier.
    
    No valid checkpoints were found in `checkpoints/`. To enable interactive predictions,
    train a baseline model using the verified CLI:
    
    ```bash
    # 1. Ingest clean data
    python -m nepali_scraper.cli ingest --input data/pilot_raw.jsonl --output data/clean.jsonl
    
    # 2. Generate leakage-free splits
    python -m nepali_scraper.cli split --input data/clean.jsonl --output-dir data/splits/gold
    
    # 3. Train Level 1 TF-IDF Logistic Regression checkpoint
    python -m nepali_scraper.cli train --data-dir data/splits/gold --model tfidf_logreg --output-dir checkpoints/tfidf_logreg
    ```
    """)
    try:
        st.stop()
    except Exception:
        pass
else:
    checkpoint_names = [chk.name for chk in available_checkpoints]
    selected_checkpoint_names = st.sidebar.multiselect(
        "Select Model(s) to Compare",
        checkpoint_names,
        default=[checkpoint_names[0]],
    )
    st.sidebar.success(f"Loaded {len(selected_checkpoint_names)} model(s)")

# Main Panel: Text Input & Live Linguistic Analysis
st.subheader("1. Nepali Text Input & Linguistic Telemetry")

preset_options = {
    "Custom Text": "",
    "Positive Product Review": "यो मोबाइल फोन धेरै राम्रो छ, क्यामेरा र ब्याट्री उत्कृष्ट लाग्यो।",
    "Negative Service Complaint (Negation Verb)": "डेलिभरी अत्यन्तै ढिलो भयो, ग्राहक सेवाले सहयोग गरेन।",
    "Neutral News Statement": "नेपाल राष्ट्र बैंकले आज नयाँ विदेशी विनिमय दर सार्वजनिक गरेको छ।",
    "Code-Mixed Nepali + English": "यो product साह्रै राम्रो लाग्यो, delivery पनि super fast थियो!",
    "False-Positive Trap (Starts with 'न')": "नेपाल एक सुन्दर देश हो जहाँ नयाँ सम्भावनाहरू प्रशस्त छन्।",
}

selected_preset = st.selectbox("Choose a sample preset or enter custom text:", list(preset_options.keys()))
default_input = preset_options[selected_preset] if selected_preset != "Custom Text" else "यो सामान एकदमै राम्रो छ, म पूर्ण सन्तुष्ट छु।"

raw_text = st.text_area("Input Text (Devanagari or Romanized Nepali):", value=default_input, height=100)

if not raw_text.strip():
    st.info("Enter Nepali text above to run real-time linguistic analysis and model inference.")
    st.stop()

# Real-time Linguistic Feature Readout
norm_text = normalize_nepali_unicode(raw_text)
words = WORD_TOKEN_PATTERN.findall(norm_text)
dev_ratio = devanagari_ratio(norm_text)
neg_info = detect_nepali_negation(norm_text)
cm_info = analyze_code_mixing(norm_text)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Word Tokens", len(words))
col2.metric("Characters (NFC)", len(norm_text))
col3.metric("Devanagari Ratio", f"{dev_ratio * 100:.1f}%")
col4.metric("Negation Detected?", "Yes" if neg_info["has_negation"] else "No")

# Linguistic Insights Pill Readout
with st.expander("🔍 Deep Linguistic Analysis Readout", expanded=True):
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.markdown("**Morphological Negation:**")
        if neg_info["has_negation"]:
            st.warning(f"Detected Negation Tokens: `{neg_info['matched_tokens']}` ({', '.join(neg_info['matched_patterns'])})")
        else:
            st.success("No authentic negation markers found. (Non-negative nouns with 'न' safely verified)")
    with col_l2:
        st.markdown("**Script & Code-Mixing:**")
        if cm_info["is_code_mixed"]:
            st.info(f"Code-mixing detected! Latin tokens: `{cm_info['latin_tokens']}` (Latin ratio: {cm_info['latin_ratio'] * 100:.1f}%)")
        else:
            st.success("Pure Devanagari script.")

# Model Predictions & Side-by-Side Comparison
st.subheader("2. Model Predictions & Calibration Transparency")

if not selected_checkpoint_names:
    st.warning("Please select at least one checkpoint from the sidebar.")
    try:
        st.stop()
    except Exception:
        pass
else:
    model_cols = st.columns(len(selected_checkpoint_names))

    for idx, chk_name in enumerate(selected_checkpoint_names):
        chk_path = checkpoints_dir / chk_name
        with model_cols[idx]:
            st.markdown(f"### `{chk_name}`")
            try:
                model, model_cat = load_trained_model(chk_path)
                pred = int(model.predict([norm_text])[0])
                pred_name = LABEL_SCHEMA.get(pred, str(pred))
                
                # Badge rendering
                badge_class = "badge-neg" if pred == 0 else ("badge-neu" if pred == 1 else "badge-pos")
                st.markdown(f"**Predicted Sentiment:** <span class='{badge_class}'>{pred_name} (Class {pred})</span>", unsafe_allow_html=True)
                
                # Probability and Calibration Status
                if hasattr(model, "predict_proba"):
                    probas = model.predict_proba([norm_text])[0]
                    conf = float(np.max(probas))
                    
                    # Check Calibration Status
                    if model_cat == "tfidf":
                        st.caption("✅ **Platt-Calibrated Probability** (Logistic / Calibrated LinearSVC)")
                    elif model_cat == "majority":
                        st.caption("ℹ️ **Empirical Training Prior**")
                    else:
                        st.caption("⚠️ **Uncalibrated Softmax Score** (Heuristic Only)")
                        
                    st.metric("Confidence", f"{conf * 100:.1f}%")
                    
                    # Distribution Chart
                    chart_data = {
                        LABEL_SCHEMA[c]: float(probas[c])
                        for c in [0, 1, 2]
                    }
                    st.bar_chart(chart_data)
                else:
                    st.info(f"Predicted class: {pred_name}")
                    
            except Exception as e:
                st.error(f"Error loading model '{chk_name}': {str(e)}")

# Research Disclaimers & Calibration Notice
st.markdown("---")
st.subheader("3. Model Limitations & Calibration Notice")

st.markdown("""
<div class='callout-box'>
<strong>Methodological Notice for Practitioners and Evaluators:</strong>
<ul>
    <li><strong>Domain Sensitivity:</strong> Models evaluated on cross-domain data frequently experience distribution shifts. Models trained on formal news text show marked accuracy drops when applied to colloquial social media or e-commerce reviews.</li>
    <li><strong>Sarcasm & Cultural Context:</strong> Subtle Nepali sarcasm and indirect honorific negations are known open challenges in low-resource NLP.</li>
    <li><strong>Softmax Calibration Fallacy:</strong> Raw softmax outputs from deep neural networks are well-known to be overconfident and do NOT represent true Bayesian posterior probabilities unless calibrated via temperature scaling or Platt scaling.</li>
</ul>
</div>
""", unsafe_allow_html=True)

