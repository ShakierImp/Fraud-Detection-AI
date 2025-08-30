# src/models/model_evaluation.py

import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)

import joblib


def evaluate_model(model, X_test, y_test, out_dir='reports/'):
    """
    Evaluate a binary classification model and generate metrics + plots.

    Parameters
    ----------
    model : estimator object
        A trained sklearn-like model with predict() and predict_proba() or decision_function().
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series or np.array
        True labels for test data.
    out_dir : str, default='reports/'
        Directory where output files (plots + metrics.json) will be saved.

    Returns
    -------
    metrics_dict : dict
        Dictionary of evaluation metrics.
    file_paths_dict : dict
        Dictionary of output file paths for plots and metrics.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Predictions
    y_pred = model.predict(X_test)

    # Probabilities or decision scores
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        raise AttributeError("Model does not support probability or decision score prediction.")

    # Metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_score)
    cm = confusion_matrix(y_test, y_pred)

    metrics_dict = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=False),
    }

    # Save metrics to JSON
    metrics_json_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(
            {k: v for k, v in metrics_dict.items() if k not in ["classification_report", "confusion_matrix"]},
            f,
            indent=4,
        )

    # --- Plots ---
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_score)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    roc_curve_path = os.path.join(out_dir, "roc_curve.png")
    plt.savefig(roc_curve_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    file_paths_dict = {
        "roc_curve": roc_curve_path,
        "confusion_matrix": cm_path,
        "metrics_json": metrics_json_path,
    }

    return metrics_dict, file_paths_dict


if __name__ == "__main__":
    # --- Demo Workflow ---
    # 1. Load test data and a trained model
    X_test = pd.read_csv("data/splits/X_test.csv")
    y_test = pd.read_csv("data/splits/y_test.csv").squeeze()
    model = joblib.load("models/random_forest_v20241027_123045.joblib")

    # 2. Evaluate the model
    metrics, file_paths = evaluate_model(model, X_test, y_test, out_dir="reports/")

    # 3. Print results
    print("Evaluation Metrics:", metrics)
    print("Generated Files:", file_paths)
