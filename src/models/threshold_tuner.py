"""
Utility for determining optimal classification thresholds based on
precision-recall tradeoffs for fraud detection.

Author: FraudGuardian AI Team
"""

import numpy as np
import json
from typing import Dict, Tuple
from sklearn.metrics import precision_recall_curve, f1_score


def tune_thresholds(y_val_true: np.ndarray, y_val_pred_proba: np.ndarray) -> Dict[str, float]:
    """
    Sweep thresholds from 0.0 to 1.0 and determine recommended cutoffs.

    Args:
        y_val_true (np.ndarray): True binary labels for validation set.
        y_val_pred_proba (np.ndarray): Predicted probabilities for positive class.

    Returns:
        Dict[str, float]: Dictionary with keys "high", "medium", "low" thresholds.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_val_true, y_val_pred_proba)

    # Add endpoints for thresholds (sklearn drops them)
    thresholds = np.append(thresholds, 1.0)

    # Compute F1 scores for all thresholds
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

    # --- Medium Risk Threshold: maximize F1 score ---
    medium_idx = np.argmax(f1_scores)
    medium_threshold = thresholds[medium_idx]

    # --- High Risk Threshold: precision >= 0.9, maximize recall ---
    high_idx = None
    for i, (p, r) in enumerate(zip(precisions, recalls)):
        if p >= 0.9:
            if high_idx is None or r > recalls[high_idx]:
                high_idx = i

    if high_idx is not None:
        high_threshold = thresholds[high_idx]
    else:
        # fallback: use medium threshold if no precision >= 0.9 found
        high_threshold = medium_threshold

    # --- Low Risk Threshold: always zero (accept everything) ---
    low_threshold = 0.0

    thresholds_dict = {
        "high": float(np.round(high_threshold, 4)),
        "medium": float(np.round(medium_threshold, 4)),
        "low": float(low_threshold),
    }

    return thresholds_dict


def save_thresholds(thresholds: Dict[str, float], path: str = "thresholds.json") -> None:
    """
    Save thresholds dictionary as JSON.

    Args:
        thresholds (Dict[str, float]): Thresholds dict from tune_thresholds.
        path (str): File path for JSON output.
    """
    with open(path, "w") as f:
        json.dump(thresholds, f, indent=4)
    print(f"[INFO] Thresholds saved to {path}")


if __name__ == "__main__":
    # Example usage with dummy validation data
    rng = np.random.RandomState(42)
    y_val_true = rng.randint(0, 2, size=1000)
    y_val_pred_proba = rng.rand(1000)

    thresholds = tune_thresholds(y_val_true, y_val_pred_proba)
    print("[INFO] Recommended thresholds:", thresholds)
    save_thresholds(thresholds)
