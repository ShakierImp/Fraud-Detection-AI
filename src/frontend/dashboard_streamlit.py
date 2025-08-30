# src/frontend/dashboard_streamlit.py
"""
Streamlit Dashboard for Local Fraud Detection Inference

Features
--------
- Loads the newest trained model artifacts (cached)
- Accepts a CSV of transactions and runs the full preprocessing + feature pipeline
- Produces fraud probabilities, risk categories, and interactive charts
- Lets you download flagged (High/Medium risk) transactions as CSV

Usage
-----
streamlit run src/frontend/dashboard_streamlit.py
"""

import os
import io
import time
import json
import pandas as pd
import streamlit as st

# Visualization libs
import matplotlib.pyplot as plt
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# Internal imports
from src.models.model_serializer import load_model_artifacts, list_models
from src.features.data_preprocessing import clean_df
from src.features.feature_engineering import get_features

# Optional anomaly scoring (Isolation Forest)
try:
    from src.models.anomaly_detection import score_anomaly  # noqa: F401
    IF_AVAILABLE = True
except Exception:
    IF_AVAILABLE = False


# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🕵️‍♂️",
    layout="wide",
)

# -----------------------------
# Helper functions
# -----------------------------
RISK_THRESHOLDS = {
    "high": 0.9,
    "medium": 0.6,
    "suspicious": 0.5,  # for the pie chart legit vs suspicious
}


def map_risk(prob: float) -> str:
    """Map probability to risk category."""
    if prob >= RISK_THRESHOLDS["high"]:
        return "High"
    elif prob >= RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "Low"


@st.cache_resource(show_spinner=False)
def load_latest_model_cached():
    """
    Load the newest model artifacts (cached for session).
    Returns a dict with keys: model, feature_list, scaler, metadata.
    """
    models = list_models()
    if not models:
        return None

    # list_models returns newest first per implementation
    newest = models[0]
    artifacts = load_model_artifacts(newest["path"])
    return artifacts


def _ensure_required_columns(df: pd.DataFrame, required_cols) -> list:
    """Return a list of missing required columns (if any)."""
    return [c for c in required_cols if c not in df.columns]


@st.cache_data(show_spinner=False)
def run_inference_pipeline(
    file_bytes: bytes,
    expected_features: list,
    model_obj,
):
    """
    Cached pipeline:
    - Read CSV bytes into DataFrame
    - Clean + feature engineering
    - Align features to model's expected feature list
    - Predict probabilities
    Returns:
        original_df (DataFrame), enhanced_df (DataFrame with probability & risk_score)
    """
    input_df = pd.read_csv(io.BytesIO(file_bytes))
    cleaned_df = clean_df(input_df)

    # Feature engineering
    X_features, _ = get_features(cleaned_df)

    # Validate and align to expected features
    missing_feats = _ensure_required_columns(X_features, expected_features)
    if missing_feats:
        raise ValueError(
            f"Missing required features for the loaded model: {missing_feats}"
        )
    X_aligned = X_features[expected_features]

    # Predict probabilities (fraud = positive class)
    if hasattr(model_obj, "predict_proba"):
        probs = model_obj.predict_proba(X_aligned)[:, 1]
    elif hasattr(model_obj, "decision_function"):
        # Convert scores to [0,1] via a simple logistic transform fallback if needed.
        # But better to require predict_proba for calibrated outputs.
        scores = model_obj.decision_function(X_aligned)
        probs = 1 / (1 + pd.np.exp(-scores))  # type: ignore[attr-defined]
    else:
        raise RuntimeError("Loaded model does not support predict_proba or decision_function.")

    enhanced_df = input_df.copy()
    enhanced_df["probability"] = probs
    enhanced_df["risk_score"] = enhanced_df["probability"].apply(map_risk)

    return input_df, enhanced_df


def render_bar_chart_risk_counts(df: pd.DataFrame):
    """Bar chart for counts per risk category."""
    counts = df["risk_score"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).astype(int)
    chart_df = counts.reset_index()
    chart_df.columns = ["risk_score", "count"]

    if PLOTLY_AVAILABLE:
        fig = px.bar(chart_df, x="risk_score", y="count", title="Transactions by Risk Category")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(chart_df["risk_score"], chart_df["count"])
        ax.set_title("Transactions by Risk Category")
        ax.set_xlabel("Risk Category")
        ax.set_ylabel("Count")
        st.pyplot(fig)


def render_pie_chart_legit_vs_suspicious(df: pd.DataFrame):
    """Pie chart: Legitimate (<0.5) vs Suspicious (>=0.5)."""
    legit = (df["probability"] < RISK_THRESHOLDS["suspicious"]).sum()
    suspicious = (df["probability"] >= RISK_THRESHOLDS["suspicious"]).sum()
    pie_df = pd.DataFrame({"label": ["Legitimate", "Suspicious"], "value": [legit, suspicious]})

    if PLOTLY_AVAILABLE:
        fig = px.pie(pie_df, names="label", values="value", title="Legitimate vs Suspicious")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(pie_df["value"], labels=pie_df["label"], autopct="%1.1f%%", startangle=90)
        ax.set_title("Legitimate vs Suspicious")
        ax.axis("equal")
        st.pyplot(fig)


# -----------------------------
# UI: Header
# -----------------------------
st.title("🕵️‍♂️ Fraud Detection Dashboard")
st.caption("Run local inference on uploaded transactions and visualize risk.")

# -----------------------------
# Load and cache model artifacts
# -----------------------------
with st.spinner("Loading latest model artifacts..."):
    ARTIFACTS = load_latest_model_cached()

if not ARTIFACTS:
    st.error("No trained models found in `models/`. Please train and save a model first.")
    st.stop()

model = ARTIFACTS["model"]
feature_list = ARTIFACTS.get("feature_list", [])
metadata = ARTIFACTS.get("metadata", {})

# Model summary
with st.expander("Model Info", expanded=False):
    st.write("**Model Name:**", metadata.get("model_name", "N/A"))
    st.write("**Version:**", metadata.get("version", "N/A"))
    st.write("**Created At:**", metadata.get("created_at", "N/A"))
    st.write("**Feature Count:**", metadata.get("feature_count", len(feature_list)))
    st.code(json.dumps(metadata, indent=2))

# -----------------------------
# File Uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a CSV file of transactions",
    type=["csv"],
    help="The file should contain columns like transaction_id, timestamp, amount, sender_id, receiver_id, sender_country, receiver_country, device_id.",
)

if uploaded_file is not None:
    try:
        # Preview first rows (without caching)
        preview_df = pd.read_csv(uploaded_file, nrows=5)
        st.subheader("Preview")
        st.dataframe(preview_df, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not preview CSV: {e}")

# Optional features
col_opt1, col_opt2 = st.columns([1, 1])
with col_opt1:
    show_charts = st.checkbox("Show charts after inference", value=True)
with col_opt2:
    enable_if_scores = st.checkbox("Compute anomaly scores (if Isolation Forest available)", value=False and IF_AVAILABLE)
    if enable_if_scores and not IF_AVAILABLE:
        st.info("Isolation Forest scoring module not available. Install/enable to use this feature.")

# -----------------------------
# Inference Button
# -----------------------------
run_btn = st.button("🚀 Run Detection", type="primary", use_container_width=True)

if run_btn:
    if uploaded_file is None:
        st.error("Please upload a CSV file first.")
        st.stop()

    try:
        file_bytes = uploaded_file.getvalue()

        # Run pipeline (cached)
        with st.spinner("Running preprocessing, feature engineering, and inference..."):
            start = time.time()
            original_df, result_df = run_inference_pipeline(
                file_bytes=file_bytes,
                expected_features=feature_list,
                model_obj=model,
            )
            elapsed = time.time() - start

        st.success(f"Inference complete in {elapsed:.2f}s")

        # Optional anomaly scores (if you have a trained IF model & want to repurpose)
        if enable_if_scores and IF_AVAILABLE:
            try:
                # Note: score_anomaly expects the features the IF model was trained on.
                # Here we demonstrate scoring directly on feature matrix from get_features.
                cleaned_df = clean_df(pd.read_csv(io.BytesIO(file_bytes)))
                X_features, _ = get_features(cleaned_df)
                # This expects you to have an IsolationForest model with scaler attached.
                # For now, we'll skip loading a separate IF model, but you can extend this easily.
                st.info("Anomaly scoring hook present, but no IsolationForest model was loaded in this dashboard.")
            except Exception as e:
                st.warning(f"Could not compute anomaly scores: {e}")

        # -----------------------------
        # Display results table
        # -----------------------------
        st.subheader("Results")
        st.dataframe(result_df, use_container_width=True)

        # -----------------------------
        # Summary Metrics
        # -----------------------------
        total_txns = len(result_df)
        flagged = (result_df["risk_score"].isin(["High", "Medium"])).sum()
        flagged_pct = (flagged / total_txns * 100) if total_txns > 0 else 0.0

        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Total Transactions", f"{total_txns:,}")
        mcol2.metric("Flagged Transactions", f"{flagged:,}")
        mcol3.metric("Flagged %", f"{flagged_pct:.1f}%")

        # -----------------------------
        # Visualizations
        # -----------------------------
        if show_charts:
            c1, c2 = st.columns(2)
            with c1:
                render_bar_chart_risk_counts(result_df)
            with c2:
                render_pie_chart_legit_vs_suspicious(result_df)

        # -----------------------------
        # Download flagged records
        # -----------------------------
        flagged_df = result_df[result_df["risk_score"].isin(["High", "Medium"])].copy()
        flagged_csv = flagged_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Flagged (High/Medium) CSV",
            data=flagged_csv,
            file_name="flagged_transactions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    except ValueError as ve:
        st.error(f"Validation error: {ve}")
    except Exception as e:
        st.exception(e)
