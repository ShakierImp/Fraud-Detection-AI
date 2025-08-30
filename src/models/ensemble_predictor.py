"""
Ensemble Predictor Utility for FraudGuardian AI

Combines supervised model probabilities with unsupervised anomaly scores
into a unified risk assessment framework.

Author: FraudGuardian AI Team
"""

import pandas as pd
import numpy as np


def _normalize_series(series: pd.Series) -> pd.Series:
    """
    Normalize a pandas Series to [0,1] using min-max scaling.
    If already within [0,1], returns unchanged.

    Args:
        series (pd.Series): Input values.

    Returns:
        pd.Series: Normalized values between 0 and 1.
    """
    if series.min() >= 0 and series.max() <= 1:
        return series.fillna(0.0)

    return (series - series.min()) / (series.max() - series.min() + 1e-8)


def ensemble_score(
    supervised_prob_series: pd.Series,
    anomaly_series: pd.Series,
    w_prob: float = 0.7,
    w_anom: float = 0.3,
) -> pd.Series:
    """
    Combine supervised model probabilities with anomaly scores into final risk scores.

    Args:
        supervised_prob_series (pd.Series): Probabilities from supervised model (0-1).
        anomaly_series (pd.Series): Anomaly scores (will be normalized to 0-1).
        w_prob (float): Weight for supervised probabilities.
        w_anom (float): Weight for anomaly scores.

    Returns:
        pd.Series: Combined risk scores in [0,1].
    """
    if not np.isclose(w_prob + w_anom, 1.0):
        raise ValueError("Weights must sum to 1.0")

    if len(supervised_prob_series) != len(anomaly_series):
        raise ValueError("Input series must have the same length")

    # Normalize anomaly scores to [0,1]
    anomaly_series = _normalize_series(anomaly_series)

    # Fill missing values
    supervised_prob_series = supervised_prob_series.fillna(0.0)
    anomaly_series = anomaly_series.fillna(0.0)

    combined = w_prob * supervised_prob_series + w_anom * anomaly_series
    combined = combined.clip(0, 1)

    return pd.Series(combined, index=supervised_prob_series.index)


def categorize_risk(
    combined_scores: pd.Series,
    thresholds: dict = None,
) -> pd.Series:
    """
    Categorize combined risk scores into Low, Medium, High.

    Args:
        combined_scores (pd.Series): Risk scores in [0,1].
        thresholds (dict, optional): Custom thresholds. Default:
            {"HIGH": 0.8, "MEDIUM": 0.4}

    Returns:
        pd.Series: Risk categories.
    """
    if thresholds is None:
        thresholds = {"HIGH": 0.8, "MEDIUM": 0.4}

    def _assign_category(score: float) -> str:
        if score >= thresholds["HIGH"]:
            return "High"
        elif score >= thresholds["MEDIUM"]:
            return "Medium"
        else:
            return "Low"

    return combined_scores.apply(_assign_category)


# ---------------- Example Usage ---------------- #
if __name__ == "__main__":
    # Example supervised probabilities (from ML model)
    probs = pd.Series([0.1, 0.5, 0.9, 0.3])

    # Example anomaly scores (from unsupervised model)
    anomalies = pd.Series([0.2, 0.6, 0.8, 0.1])

    # Combine scores
    combined = ensemble_score(probs, anomalies, w_prob=0.7, w_anom=0.3)
    print("[INFO] Combined Scores:\n", combined)

    # Categorize risk
    risk_levels = categorize_risk(combined, thresholds={"HIGH": 0.85, "MEDIUM": 0.5})
    print("\n[INFO] Risk Levels:\n", risk_levels)
