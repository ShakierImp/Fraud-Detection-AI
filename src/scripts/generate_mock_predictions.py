# src/scripts/generate_mock_predictions.py
# -------------------------------------------------------------------
# Script: Generate deterministic, realistic-looking mock predictions
# Purpose: Create demo fraud prediction outputs for FraudGuardian AI
#
# Input:  data/sample_input.csv
# Output: data/demo_predictions.csv
#
# Columns:
#   - transaction_id (from original input)
#   - probability (fraud likelihood, float 0-1)
#   - risk_score ("Low", "Medium", "High")
#   - anomaly_score (float 0-1, correlated with probability)
#
# Notes:
#   - Deterministic generation based on transaction_id hash
#   - Distribution: ~80-90% Low, 8-15% Medium, 2-5% High
# -------------------------------------------------------------------

import os
import hashlib
import pandas as pd
import numpy as np
from src.utils.config import DATA_DIR


def deterministic_float(value: str, seed: str, scale: int = 1000) -> float:
    """
    Generate a deterministic float between 0-1 based on a string value and seed.
    """
    key = f"{value}_{seed}".encode("utf-8")
    h = hashlib.md5(key).hexdigest()
    return (int(h, 16) % scale) / float(scale)


def assign_risk(probability: float) -> str:
    """
    Assign risk score based on probability thresholds.
    """
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"


def main():
    print("🔹 Generating mock predictions for demo...")

    input_path = os.path.join(DATA_DIR, "sample_input.csv")
    output_path = os.path.join(DATA_DIR, "demo_predictions.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ Input file not found: {input_path}")

    # Load input
    df = pd.read_csv(input_path)
    if "transaction_id" not in df.columns:
        raise ValueError("❌ Input CSV must contain 'transaction_id' column")

    # Generate mock predictions
    probs = []
    risks = []
    anomalies = []

    for tid in df["transaction_id"].astype(str):
        prob = deterministic_float(tid, seed="prob", scale=1000)

        # Skew distribution to match requirements
        if prob < 0.8:
            prob *= 0.3  # majority low risk
        elif prob < 0.95:
            prob = 0.3 + (prob - 0.8) * (0.4 / 0.15)  # medium range
        else:
            prob = 0.7 + (prob - 0.95) * (0.3 / 0.05)  # high range

        risk = assign_risk(prob)
        anomaly = min(1.0, max(0.0, prob + deterministic_float(tid, seed="anom", scale=200) - 0.5))

        probs.append(round(prob, 4))
        risks.append(risk)
        anomalies.append(round(anomaly, 4))

    out_df = pd.DataFrame({
        "transaction_id": df["transaction_id"],
        "probability": probs,
        "risk_score": risks,
        "anomaly_score": anomalies
    })

    # Save to CSV
    out_df.to_csv(output_path, index=False)
    print(f"✅ Mock predictions saved to {output_path}")

    # Print summary statistics
    summary = out_df["risk_score"].value_counts(normalize=True).mul(100).round(2)
    print("\n📊 Risk Score Distribution (%):")
    print(summary.to_string())

    print("\n🔎 Example rows:")
    print(out_df.head())


if __name__ == "__main__":
    main()
