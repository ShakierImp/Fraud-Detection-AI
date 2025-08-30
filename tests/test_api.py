# tests/test_api.py
"""
Integration tests for FastAPI endpoints in the fraud detection project.

Covers:
- GET /health
- POST /predict (CSV and JSON input)
"""

import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient

# Import the FastAPI app
from src.api.inference_api_fast import app


# -------------------
# Fixtures
# -------------------

@pytest.fixture(scope="module")
def client():
    """Fixture to provide a FastAPI TestClient for the app."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_csv_file():
    """Fixture that provides path to sample CSV input."""
    csv_path = "data/sample_input.csv"
    if not os.path.exists(csv_path):
        pytest.skip(f"Sample CSV not found at {csv_path}. Skipping CSV tests.")
    return csv_path


@pytest.fixture(scope="module")
def sample_json_data():
    """Fixture that provides example transaction JSON data."""
    return [
        {
            "transaction_id": "txn_001",
            "user_id": "user_001",
            "txn_amount": 123.45,
            "txn_time": "2025-08-28 10:15:00",
            "user_location": "NY",
            "merchant_location": "CA",
        },
        {
            "transaction_id": "txn_002",
            "user_id": "user_002",
            "txn_amount": 67.89,
            "txn_time": "2025-08-28 11:45:00",
            "user_location": "TX",
            "merchant_location": "TX",
        },
    ]


# -------------------
# Tests
# -------------------

def test_health_endpoint_returns_200(client):
    """Test that /health returns 200 and correct JSON structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert "model_loaded" in data
    assert isinstance(data["model_loaded"], bool)
    assert data["model_loaded"] is True


def test_predict_csv_returns_200(client, sample_csv_file):
    """Test that /predict accepts CSV and returns predictions + summary."""
    with open(sample_csv_file, "rb") as f:
        response = client.post("/predict", files={"file": ("sample.csv", f, "text/csv")})
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert "summary" in data
    assert isinstance(data["summary"], dict)


def test_predict_json_returns_200(client, sample_json_data):
    """Test that /predict accepts JSON payload and returns predictions."""
    response = client.post("/predict", json=sample_json_data)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert "summary" in data


def test_predict_without_file_returns_error(client):
    """Test that /predict without any input returns validation error."""
    response = client.post("/predict")
    assert response.status_code in [400, 422]  # FastAPI may return 422 Unprocessable Entity
