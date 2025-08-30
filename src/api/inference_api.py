"""
src/api/inference_api.py

Production-ready Flask API for serving fraud detection predictions.
"""

import os
import json
import pandas as pd
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest, InternalServerError

from src.models.model_serializer import load_model_artifacts, list_models
from src.features.data_preprocessing import clean_df
from src.features.feature_engineering import get_features

# -------------------------------------------------
# Global variables to hold loaded model artifacts
# -------------------------------------------------
ARTIFACTS = {
    "model": None,
    "feature_list": None,
    "scaler": None,
    "metadata": None
}

# -------------------------------------------------
# Flask app initialization
# -------------------------------------------------
app = Flask(__name__)


def load_latest_model():
    """Load the newest available model artifacts into global ARTIFACTS."""
    global ARTIFACTS
    models = list_models()
    if not models:
        app.logger.warning("No models found in base_dir.")
        return False

    latest_model_dir = models[0]["path"]
    artifacts = load_model_artifacts(latest_model_dir)
    ARTIFACTS.update(artifacts)
    app.logger.info(f"Loaded model from {latest_model_dir}")
    return True


# Load model at startup
MODEL_LOADED = load_latest_model()


# -------------------------------------------------
# Utility functions
# -------------------------------------------------
def map_risk(prob: float) -> str:
    """Map probability to risk category."""
    if prob >= 0.9:
        return "High"
    elif prob >= 0.6:
        return "Medium"
    else:
        return "Low"


from flask import Request

def prepare_input_dataframe(req: Request) -> pd.DataFrame:
    """Parse incoming request data into a pandas DataFrame."""
    if req.content_type.startswith("multipart/form-data"):
        file = req.files.get("file")
        if not file or not file.filename:
            raise BadRequest("No file part in the request.")
        return pd.read_csv(file)

    elif req.is_json:
        data = req.get_json()
        if not isinstance(data.get("transactions"), list):
            raise BadRequest("JSON body must contain a 'transactions' list.")
        return pd.DataFrame(data["transactions"])

    else:
        raise BadRequest("Unsupported content type. Use CSV (multipart/form-data) or JSON (application/json).")
# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL_LOADED
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Fraud detection prediction endpoint.

    Example curl commands:

    # JSON input
    curl -X POST http://localhost:5000/predict \
        -H "Content-Type: application/json" \
        -d '{"transactions": [{"transaction_id": 1, "amount": 123.45, "feature_x": 10}, {"transaction_id": 2, "amount": 67.89, "feature_x": 5}]}'

    # CSV input
    curl -X POST http://localhost:5000/predict \
        -F "file=@/path/to/transactions.csv"
    """
    if not MODEL_LOADED:
        raise InternalServerError("No model loaded. Cannot perform predictions.")

    # Step 1: Parse input
    input_df = prepare_input_dataframe(request)
    if input_df.empty:
        raise BadRequest("Input data is empty.")

    # Step 2: Clean + Feature engineering
    cleaned_df = clean_df(input_df)
    X_features, _ = get_features(cleaned_df)

    # Ensure feature consistency
    expected_features = ARTIFACTS["feature_list"]
    missing = [f for f in expected_features if f not in X_features.columns]
    if missing:
        raise BadRequest(f"Missing required features: {missing}")
    X_aligned = X_features[expected_features]

    # Step 3: Prediction
    model = ARTIFACTS["model"]
    probs = model.predict_proba(X_aligned)[:, 1]

    # Step 4: Build response
    predictions = []
    summary = {"High": 0, "Medium": 0, "Low": 0}

    for i, prob in enumerate(probs):
        transaction_id = input_df.iloc[i].get("transaction_id", i)
        risk = map_risk(prob)
        summary[risk] += 1
        predictions.append({
            "transaction_id": transaction_id,
            "probability": float(prob),
            "risk_score": risk
        })

    return jsonify({
        "predictions": predictions,
        "summary": summary,
        "model_version": ARTIFACTS["metadata"].get("version") if ARTIFACTS["metadata"] else None
    })


# -------------------------------------------------
# Error Handlers
# -------------------------------------------------
@app.errorhandler(BadRequest)
def handle_bad_request(e):
    return jsonify(error="Bad Request", message=str(e)), 400


@app.errorhandler(InternalServerError)
def handle_internal_error(e):
    return jsonify(error="Internal Server Error", message=str(e)), 500


@app.errorhandler(Exception)
def handle_unexpected_error(e):
    return jsonify(error="Unexpected Error", message=str(e)), 500


# -------------------------------------------------
# CLI Demo
# -------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
