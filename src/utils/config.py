# src/utils/config.py
# ------------------------------------------------------------
# Centralized configuration module for FraudGuardian AI
# Stores constants, paths, hyperparameters, API settings,
# and feature engineering defaults for consistency across project
# ------------------------------------------------------------

"""
This module centralizes all configuration settings for the FraudGuardian AI project.
It includes directory paths, default hyperparameters, API server settings,
risk scoring thresholds, and feature engineering constants.

Usage Example:
--------------
from src.utils.config import MODEL_DIR, RISK_THRESHOLDS, DEFAULT_RANDOM_STATE

print(f"Models stored in: {MODEL_DIR}")
print(f"High risk threshold: {RISK_THRESHOLDS['HIGH']}")
"""

import os

# ------------------------------------------------------------
# Paths & Directories (relative to project root for portability)
# ------------------------------------------------------------
MODEL_DIR: str = os.path.join("models/")
"""Directory for storing trained ML models."""

DATA_DIR: str = os.path.join("data/")
"""Root directory for datasets."""

SPLITS_DIR: str = os.path.join(DATA_DIR, "splits/")
"""Directory for train/validation/test split files."""

LOG_DIR: str = os.path.join("logs/")
"""Directory for log files."""

TUNING_DIR: str = os.path.join("tuning/")
"""Directory for hyperparameter tuning results."""


# ------------------------------------------------------------
# Model Parameters (default hyperparameters & reproducibility)
# ------------------------------------------------------------
DEFAULT_CONTAMINATION: float = 0.01
"""Default contamination parameter for anomaly detection models (e.g., Isolation Forest)."""

DEFAULT_RANDOM_STATE: int = 42
"""Random seed for reproducibility across experiments."""

DEFAULT_TEST_SIZE: float = 0.2
"""Proportion of dataset reserved for testing."""

DEFAULT_VAL_SIZE: float = 0.1
"""Proportion of dataset reserved for validation."""


# ------------------------------------------------------------
# Risk Thresholds (used in scoring and classification)
# ------------------------------------------------------------
RISK_THRESHOLDS: dict = {
    "HIGH": 0.9,
    "MEDIUM": 0.6,
    "LOW": 0.0,
}
"""Risk categories and their minimum threshold scores."""

ANOMALY_THRESHOLD: float = 0.8
"""Threshold for anomaly detection using reconstruction errors (e.g., autoencoder)."""


# ------------------------------------------------------------
# API Configuration (server ports & settings)
# ------------------------------------------------------------
API_HOST: str = "0.0.0.0"
"""Host address for running the FastAPI backend."""

API_PORT: int = 8000
"""Port for serving the API."""

STREAMLIT_PORT: int = 8501
"""Port for running the Streamlit dashboard."""


# ------------------------------------------------------------
# Feature Constants (dataset schema & feature engineering)
# ------------------------------------------------------------
TARGET_COLUMN: str = "label"
"""Name of the target column for classification (fraud = 1, non-fraud = 0)."""

TIMESTAMP_COLUMN: str = "timestamp"
"""Name of the timestamp column in the dataset."""

AMOUNT_COLUMN: str = "amount"
"""Name of the transaction amount column in the dataset."""


# ------------------------------------------------------------
# Utility: Ensure critical directories exist
# ------------------------------------------------------------
for directory in [MODEL_DIR, DATA_DIR, SPLITS_DIR, LOG_DIR, TUNING_DIR]:
    os.makedirs(directory, exist_ok=True)
