# src/models/anomaly_detection.py
"""
Unsupervised fraud detection with Isolation Forest.

This module provides a production-style workflow to:
- Fit an Isolation Forest (IF) on unlabeled, numeric training features.
- Produce anomaly scores in [0, 1], where 1 == most anomalous.
- Persist the trained model (including its score scaler) to disk.

Design principles:
- Immutability: Inputs are never modified in-place.
- Reproducibility: Random states are honored.
- Practicality: Scaler for scores is fitted on training predictions and stored
  on the model object so subsequent scoring uses the same calibration.

Usage:
    python -m src.models.anomaly_detection
"""

from __future__ import annotations

from typing import Optional

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


def fit_isolation_forest(
    X_train: pd.DataFrame,
    contamination: float = 0.01,
    random_state: int = 42,
) -> IsolationForest:
    """Fit an Isolation Forest model on training data and attach a score scaler.

    The scaler is fit on the negative of the Isolation Forest decision function
    (so that larger numbers indicate higher anomaly). It is stored on the model
    as `score_scaler_` and will be used by :func:`score_anomaly`.

    Args:
        X_train (pd.DataFrame): Training features (numeric). The DataFrame is not modified.
        contamination (float, optional): Expected fraction of outliers. Defaults to 0.01.
        random_state (int, optional): RNG seed for reproducibility. Defaults to 42.

    Returns:
        IsolationForest: A fitted Isolation Forest model with an attached
            attribute `score_scaler_` (MinMaxScaler fitted on training scores)
            and `contamination` reflecting the configuration.
    """
    # Defensive copy (immutability of caller's object)
    X_mat = X_train.copy()

    # Fit IF
    if_model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
        bootstrap=False,
        verbose=0,
    )
    if_model.fit(X_mat)

    # Fit score scaler on training predictions
    # IsolationForest.decision_function: larger = more "normal", more negative = more "anomalous"
    raw_scores = if_model.decision_function(X_mat)  # shape (n_samples,)
    neg_scores = -raw_scores.reshape(-1, 1)  # invert so larger => more anomalous

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(neg_scores)

    # Attach scaler and configuration attributes to the model
    setattr(if_model, "score_scaler_", scaler)
    setattr(if_model, "contamination", contamination)

    return if_model


def score_anomaly(model: IsolationForest, X: pd.DataFrame) -> pd.Series:
    """Compute normalized anomaly scores in [0, 1] for new data.

    The function uses the fitted IsolationForest's decision function and the
    MinMaxScaler that was fit on training scores inside :func:`fit_isolation_forest`.
    By convention here, 1.0 corresponds to the most anomalous points.

    Behavior:
        - If the model does not have `score_scaler_`, the function will fit a
          new MinMaxScaler on the provided data's scores (not the preferred
          practice), and attach it to the model for consistency.

    Args:
        model (IsolationForest): A fitted Isolation Forest model returned by
            :func:`fit_isolation_forest`. Should carry `score_scaler_`.
        X (pd.DataFrame): Feature matrix to score. Not modified.

    Returns:
        pd.Series: Anomaly scores in [0, 1], indexed identically to `X`,
            with name 'anomaly_score' where higher values indicate higher anomaly.
    """
    X_mat = X.copy()

    # Compute raw IF scores: higher = more normal; more negative = more anomalous
    raw_scores = model.decision_function(X_mat)  # (n_samples,)
    neg_scores = -raw_scores.reshape(-1, 1)

    # Use stored scaler if available; otherwise fit on-the-fly (fallback)
    scaler: Optional[MinMaxScaler] = getattr(model, "score_scaler_", None)
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        scaler.fit(neg_scores)
        setattr(model, "score_scaler_", scaler)

    scaled = scaler.transform(neg_scores).ravel()  # (n_samples,)

    return pd.Series(scaled, index=X.index, name="anomaly_score")


def save_if_model(model: IsolationForest, path: str) -> None:
    """Persist a fitted Isolation Forest model to disk via joblib.

    Creates the parent directory if it does not exist.

    Args:
        model (IsolationForest): Fitted model to save (with attached scaler).
        path (str): Destination file path (e.g., 'models/isolation_forest.joblib').

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)


# -----------------------------------------------------------------------------
# CLI Demo
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Load pre-split training data (features only)
    X_train_path = "data/splits/X_train.csv"
    if not os.path.exists(X_train_path):
        raise FileNotFoundError(
            f"Could not find '{X_train_path}'. "
            "Ensure you have generated splits before running this demo."
        )

    X_train = pd.read_csv(X_train_path)

    # 2) Fit the Isolation Forest model (with score scaler attached)
    if_model = fit_isolation_forest(X_train, contamination=0.01, random_state=42)

    # 3) Score the training data itself (demo)
    scores = score_anomaly(if_model, X_train)

    # 4) Print stats
    print(f"Fitted Isolation Forest with contamination={if_model.contamination}")
    print(
        "Anomaly scores for training data: "
        f"min={scores.min():.3f}, max={scores.max():.3f}, mean={scores.mean():.3f}"
    )
    print(scores.head())

    # 5) Save the model
    out_path = "models/isolation_forest.joblib"
    save_if_model(if_model, out_path)
    print(f"Model saved to: {out_path}")
