# src/api/inference_api_fast.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Literal
from fastapi.testclient import TestClient
import pandas as pd
import uvicorn
import os
import io

# Internal imports
from src.models.model_serializer import load_model_artifacts, list_models
from src.features.data_preprocessing import clean_df
from src.features.feature_engineering import get_features
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# Pydantic Models
# -----------------------

class Transaction(BaseModel):
    """
    Schema for a single transaction input.
    """
    transaction_id: str = Field(..., example="txn_12345")
    timestamp: str = Field(..., example="2025-08-26 14:23:01")
    amount: float = Field(..., example=250.75)
    sender_id: str = Field(..., example="user_001")
    receiver_id: str = Field(..., example="merchant_123")
    sender_country: str = Field(..., example="US")
    receiver_country: str = Field(..., example="IN")
    device_id: str = Field(..., example="device_456")


class PredictionResult(BaseModel):
    """
    Schema for a single prediction result.
    """
    transaction_id: str
    probability: float
    risk_score: Literal['High', 'Medium', 'Low']


class PredictResponse(BaseModel):
    """
    Response schema containing predictions and summary statistics.
    """
    predictions: List[PredictionResult]
    summary: Dict[str, int]


# -----------------------
# FastAPI App
# -----------------------

app = FastAPI(title="Fraud Detection API", version="1.0.0", description="Serve fraud detection predictions via FastAPI.")


@app.on_event("startup")
async def load_model_on_startup():
    """
    Load the newest model artifacts at application startup.
    """
    try:
        models = list_models()
        if not models:
            logger.warning("No models found. API will run without a model.")
            app.state.model_artifacts = None
            return
            
        latest_model = models[0]  # list_models returns newest first
        artifacts = load_model_artifacts(latest_model["path"])
        app.state.model_artifacts = artifacts
        logger.info(f"Loaded model: {latest_model}")
    except Exception as e:
        app.state.model_artifacts = None
        logger.error(f"Failed to load model: {e}")


# -----------------------
# Helper Functions
# -----------------------

def risk_category(prob: float) -> str:
    """Categorize risk based on probability."""
    if prob >= 0.9:
        return "High"
    elif prob >= 0.6:
        return "Medium"
    else:
        return "Low"


def run_inference(df: pd.DataFrame) -> PredictResponse:
    """
    Clean, transform, and run predictions on the input DataFrame.
    """
    if app.state.model_artifacts is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    model = app.state.model_artifacts["model"]
    feature_list = app.state.model_artifacts["feature_list"]

    # Preprocessing
    cleaned_df = clean_df(df)
    X_features, _ = get_features(cleaned_df, feature_list=feature_list)
    
    # Ensure we have the required features
    missing_features = [f for f in feature_list if f not in X_features.columns]
    if missing_features:
        # Add missing features with default values
        for feature in missing_features:
            X_features[feature] = 0
    
    # Select only the features the model expects
    X_features = X_features[feature_list]

    # Predictions - handle case where model might not have predict_proba
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_features)[:, 1]
    else:
        # Fallback to binary predictions converted to probabilities
        preds = model.predict(X_features)
        probs = preds.astype(float)

    predictions = []
    summary = {"High": 0, "Medium": 0, "Low": 0}

    # Handle case where transaction_id might not exist
    if "transaction_id" in cleaned_df.columns:
        transaction_ids = cleaned_df["transaction_id"]
    else:
        transaction_ids = [f"txn_{i}" for i in range(len(cleaned_df))]
    
    for tid, prob in zip(transaction_ids, probs):
        risk = risk_category(prob)
        predictions.append(PredictionResult(transaction_id=tid, probability=float(prob), risk_score=risk))
        summary[risk] += 1

    return PredictResponse(predictions=predictions, summary=summary)


# -----------------------
# Endpoints
# -----------------------

@app.get("/health")
async def health():
    """
    Health check endpoint.
    Returns model load status.
    """
    return {"status": "ok", "model_loaded": app.state.model_artifacts is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(None), transactions: List[Transaction] = None):
    """
    Fraud prediction endpoint.
    
    Accepts either:
    - A CSV file uploaded under 'file' (multipart/form-data).
    - A JSON array of transactions.

    Example curl (CSV):
    ```bash
    curl -X POST "http://localhost:8000/predict" -F "file=@transactions.csv"
    ```

    Example curl (JSON):
    ```bash
    curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" \
         -d '{"transactions":[{"transaction_id":"txn_123","timestamp":"2025-08-26 14:23:01","amount":250.75,"sender_id":"user_1","receiver_id":"merchant_2","sender_country":"US","receiver_country":"IN","device_id":"dev_1"}]}'
    ```
    """
    try:
        if file is not None:
            contents = await file.read()
            df = pd.read_csv(io.BytesIO(contents))
        elif transactions is not None:
            df = pd.DataFrame([t.dict() for t in transactions])
        else:
            raise HTTPException(status_code=400, detail="No valid input provided. Upload a CSV file or provide JSON.")

        if "transaction_id" not in df.columns:
            raise HTTPException(status_code=400, detail="Missing required column: transaction_id")

        return run_inference(df)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------
# Testing with TestClient
# -----------------------

def test_api_with_client():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    print("Health check:", response.json())


# -----------------------
# Run with Uvicorn
# -----------------------

if __name__ == "__main__":
    uvicorn.run("src.api.inference_api_fast:app", host="0.0.0.0", port=8000, reload=True)
