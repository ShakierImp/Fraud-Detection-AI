"""
model_training.py

Production-style script for training and evaluating Random Forest and XGBoost models
on a binary classification task (fraud detection). Focuses on handling class imbalance
and saving trained models with timestamped filenames.
"""

import os
import joblib
import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier


def train_random_forest(X_train, y_train, X_val, y_val, params=None):
    """
    Train a Random Forest classifier with class imbalance handling.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation labels.
        params (dict, optional): Hyperparameters for RandomForestClassifier. 
                                 If None, sensible defaults are used.

    Returns:
        tuple: (model, metrics_dict)
            - model (RandomForestClassifier): Trained Random Forest model.
            - metrics_dict (dict): Dictionary of validation metrics:
                {"precision": float, "recall": float, "f1": float, "roc_auc": float}
    """
    # Handle case where X_train and X_val might be numpy arrays
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(X_val, 'values'):
        X_val = X_val.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    if hasattr(y_val, 'values'):
        y_val = y_val.values
    
    if params is None:
        params = {
            "n_estimators": 100,
            "random_state": 42,
            "class_weight": "balanced",
            "n_jobs": -1,
        }
    else:
        params["class_weight"] = "balanced"  # enforce class imbalance handling

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    # Metrics
    metrics_dict = {
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val, y_proba),
    }

    print("\nRandom Forest Validation Metrics:")
    for k, v in metrics_dict.items():
        print(f"{k}: {v:.4f}")

    return model, metrics_dict


def train_xgboost(X_train, y_train, X_val, y_val, params=None):
    """
    Train an XGBoost classifier with class imbalance handling.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation labels.
        params (dict, optional): Hyperparameters for XGBClassifier. 
                                 If None, sensible defaults are used.

    Returns:
        tuple: (model, metrics_dict)
            - model (XGBClassifier): Trained XGBoost model.
            - metrics_dict (dict): Dictionary of validation metrics:
                {"precision": float, "recall": float, "f1": float, "roc_auc": float}
    """
    # Compute imbalance ratio
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    if params is None:
        params = {
            "n_estimators": 100,
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "scale_pos_weight": scale_pos_weight,
            "n_jobs": -1,
        }
    else:
        params["scale_pos_weight"] = scale_pos_weight  # enforce imbalance handling

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    # Metrics
    metrics_dict = {
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val, y_proba),
    }

    print("\nXGBoost Validation Metrics:")
    for k, v in metrics_dict.items():
        print(f"{k}: {v:.4f}")

    return model, metrics_dict


def save_model(model, model_name, base_path="models/"):
    """
    Save a trained model to disk with a timestamped filename.

    Args:
        model (object): Trained model object.
        model_name (str): Short name of the model (e.g., "random_forest", "xgboost").
        base_path (str): Directory where the model will be saved.

    Returns:
        str: Full file path of the saved model.
    """
    os.makedirs(base_path, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_v{timestamp}.joblib"
    full_path = os.path.join(base_path, filename)

    joblib.dump(model, full_path)
    print(f"Model saved to: {full_path}")

    return full_path


if __name__ == "__main__":
    # CLI Demo Workflow
    print("=== Fraud Detection Model Training ===")

    # 1. Load pre-split data
    X_train = pd.read_csv("data/splits/X_train.csv")
    y_train = pd.read_csv("data/splits/y_train.csv").squeeze()
    X_val = pd.read_csv("data/splits/X_val.csv")
    y_val = pd.read_csv("data/splits/y_val.csv").squeeze()

    # 2. Train Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)
    rf_path = save_model(rf_model, "random_forest")

    # 3. Train XGBoost
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_path = save_model(xgb_model, "xgboost")

    # 4. Print summary
    print("\n=== Training Summary ===")
    print(f"Random Forest saved to: {rf_path}")
    print(f"XGBoost saved to: {xgb_path}")
