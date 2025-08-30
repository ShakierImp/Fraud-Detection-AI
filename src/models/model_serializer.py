# src/models/model_serializer.py

import os
import json
import joblib
from datetime import datetime


def save_model_artifacts(model, feature_list, model_name='model', scaler=None, base_dir='models/') -> dict:
    """
    Save model artifacts (model, features, scaler, metadata) with versioning.

    Args:
        model: Trained ML model to be saved.
        feature_list (list): List of features the model was trained on.
        model_name (str): Name of the model.
        scaler: Optional scaler object to save.
        base_dir (str): Base directory where model versions are stored.

    Returns:
        dict: Metadata dictionary.
    """
    # Generate version
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(base_dir, f"{model_name}_v{version}")
    os.makedirs(model_dir, exist_ok=True)

    # File paths
    model_path = os.path.join(model_dir, "model.joblib")
    features_path = os.path.join(model_dir, "features.json")
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    metadata_path = os.path.join(model_dir, "metadata.json")

    # Save artifacts
    joblib.dump(model, model_path)
    with open(features_path, "w") as f:
        json.dump(feature_list, f, indent=4)

    scaler_saved = False
    if scaler is not None:
        joblib.dump(scaler, scaler_path)
        scaler_saved = True

    metadata = {
        "model_name": model_name,
        "version": version,
        "created_at": datetime.now().isoformat(),
        "feature_count": len(feature_list),
        "scaler_saved": scaler_saved,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return metadata


def load_model_artifacts(model_dir_path) -> dict:
    """
    Load model artifacts from a given directory.

    Args:
        model_dir_path (str): Path to the model version directory.

    Returns:
        dict: Loaded artifacts including model, features, scaler, and metadata.
    """
    model_path = os.path.join(model_dir_path, "model.joblib")
    features_path = os.path.join(model_dir_path, "features.json")
    scaler_path = os.path.join(model_dir_path, "scaler.joblib")
    metadata_path = os.path.join(model_dir_path, "metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found in {model_dir_path}")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in {model_dir_path}")

    # Load artifacts
    model = joblib.load(model_path)
    with open(features_path, "r") as f:
        feature_list = json.load(f)

    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return {
        "model": model,
        "feature_list": feature_list,
        "scaler": scaler,
        "metadata": metadata,
    }


def list_models(base_dir='models/') -> list:
    """
    List all available model versions in the base directory.

    Args:
        base_dir (str): Base directory containing model versions.

    Returns:
        list: Sorted list (newest first) of model directories containing metadata.
    """
    if not os.path.exists(base_dir):
        return []

    model_dirs = []
    for entry in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, entry)
        metadata_path = os.path.join(dir_path, "metadata.json")
        if os.path.isdir(dir_path) and os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            model_dirs.append({
                "path": dir_path,
                "created_at": metadata.get("created_at", ""),
                "model_name": metadata.get("model_name", ""),
                "version": metadata.get("version", "")
            })

    # Sort by created_at (descending)
    model_dirs_sorted = sorted(
        model_dirs,
        key=lambda x: x.get("created_at", ""),
        reverse=True
    )

    return model_dirs_sorted


if __name__ == "__main__":
    # CLI demo
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    print("=== Model Serializer Demo ===")

    # 1. Create dummy artifacts
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)
    dummy_model = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)
    dummy_scaler = StandardScaler().fit(X)
    feature_list = ['ftr1', 'ftr2', 'ftr3', 'ftr4', 'ftr5']

    # 2. Save the artifacts
    meta = save_model_artifacts(dummy_model, feature_list, 'test_model', dummy_scaler)
    print("Saved model metadata:", meta)

    # 3. List available models
    available_models = list_models()
    print("Available models:", available_models)

    # 4. Load the artifacts back
    if available_models:
        latest_model_path = available_models[0]['path']
        artifacts = load_model_artifacts(latest_model_path)
        print("Loaded keys:", artifacts.keys())
        print("Model type:", type(artifacts['model']))
        print("Features:", artifacts['feature_list'])
