# src/scripts/train_quick_demo.py
# ------------------------------------------------------------
# Quick end-to-end training demo for FraudGuardian AI
# ------------------------------------------------------------
# This script demonstrates a lightweight pipeline:
# - Load sample dataset
# - Clean & preprocess data
# - Extract features
# - Train a small Random Forest model
# - Save model to disk for later use
#
# Designed to run quickly (<2 minutes) for testing setup
# ------------------------------------------------------------

import os
import sys
import traceback

from src.data_collection import get_sample_df
from src.features.data_preprocessing import clean_df
from src.features.feature_engineering import get_features
from src.models.model_training import train_random_forest, save_model
from src.utils.config import MODEL_DIR


def main():
    """Run a lightweight demo training pipeline."""
    try:
        print("🔹 Starting FraudGuardian quick training demo...")

        # ------------------------------------------------------------
        # 1. Load sample data
        # ------------------------------------------------------------
        print("📥 Loading sample dataset...")
        df = get_sample_df()
        print(f"✅ Loaded dataset with shape: {df.shape}")

        # ------------------------------------------------------------
        # 2. Clean & preprocess data
        # ------------------------------------------------------------
        print("🧹 Cleaning and preprocessing data...")
        df_clean = clean_df(df)
        print(f"✅ Data after cleaning: {df_clean.shape}")

        # ------------------------------------------------------------
        # 3. Extract features
        # ------------------------------------------------------------
        print("⚙️ Extracting features...")
        X, y = get_features(df_clean)
        print(f"✅ Features extracted: X={X.shape}, y={y.shape}")

        # ------------------------------------------------------------
        # 4. Train Random Forest (lightweight config)
        # ------------------------------------------------------------
        print("🌲 Training lightweight Random Forest model...")
        model = train_random_forest(
            X, y,
            n_estimators=10,
            max_depth=5,
            random_state=42
        )
        print("✅ Model training complete")

        # ------------------------------------------------------------
        # 5. Save model
        # ------------------------------------------------------------
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, "demo_model.joblib")

        print(f"💾 Saving model to {model_path}...")
        save_model(model, model_path)
        print(f"✅ Model successfully saved at: {model_path}")

        print("🎉 Quick training demo finished successfully!")

    except Exception as e:
        print("❌ An error occurred during the quick demo pipeline.")
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
