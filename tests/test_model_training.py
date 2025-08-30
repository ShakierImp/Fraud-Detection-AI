# tests/test_model_training.py
# -------------------------------------------------------------------
# Smoke test for model training pipeline in FraudGuardian AI
#
# Purpose:
#   - Ensure RandomForest training runs end-to-end without errors
#   - Verify model + metrics outputs
#   - Save/load trained model successfully
#   - Clean up after test
# -------------------------------------------------------------------

import os
import tempfile
import numpy as np
import joblib
import pytest

# CORRECTED IMPORT - changed from train_model to model_training
from src.models.model_training import train_random_forest


@pytest.mark.smoke
def test_train_random_forest_smoke():
    """
    Smoke test: Train RandomForest on a tiny synthetic dataset and verify outputs.
    """

    # --- Arrange ---
    np.random.seed(42)  # reproducibility
    X_train = np.random.rand(10, 5)  # 10 samples, 5 features
    y_train = np.random.randint(0, 2, 10)  # binary labels

    # temporary file for model
    tmp_dir = tempfile.mkdtemp()
    model_path = os.path.join(tmp_dir, "test_model.joblib")

    # --- Act ---
    model, metrics = train_random_forest(
        X_train,
        y_train,
        X_train,  # Using same data for val as train for smoke test
        y_train,
        params={"n_estimators": 5, "random_state": 42},
    )

    # Save the model manually since the function might not have save parameter
    joblib.dump(model, model_path)

    # --- Assert ---
    # model object returned
    assert model is not None, "Model training returned None"

    # metrics contains required keys
    expected_keys = {"precision", "recall", "f1", "roc_auc"}
    assert expected_keys.issubset(metrics.keys()), f"Missing keys in metrics: {metrics.keys()}"

    # model file created
    assert os.path.exists(model_path), f"Model file not found at {model_path}"

    # model can be loaded back
    loaded_model = joblib.load(model_path)
    assert loaded_model is not None, "Failed to load saved model"

    # --- Cleanup ---
    os.remove(model_path)
    os.rmdir(tmp_dir)


# Add this if you want to run the test directly
if __name__ == "__main__":
    test_train_random_forest_smoke()
    print("✅ Smoke test passed!")