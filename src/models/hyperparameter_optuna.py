# src/models/hyperparameter_optuna.py
# -------------------------------------------------------------------
# Hyperparameter tuning for XGBoost using Optuna
# -------------------------------------------------------------------
# This script demonstrates how to use Optuna to optimize key parameters
# of an XGBoost classifier for fraud detection tasks.
#
# Why Optuna?
# Optuna provides efficient hyperparameter optimization using
# techniques like TPE (Tree-structured Parzen Estimator).
# This is particularly useful for fraud detection, where model
# performance is highly sensitive to the right parameter settings.
#
# Key Metric:
# We use F1 score for evaluation since fraud detection datasets are
# often highly imbalanced. F1 balances precision and recall.
# -------------------------------------------------------------------

import os
import json
import joblib
import optuna
import xgboost as xgb
import pandas as pd
from sklearn.metrics import f1_score


def objective(trial, X_train, y_train, X_val, y_val):
    """
    Objective function for Optuna.
    Defines the hyperparameter search space and evaluates model performance.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object used to suggest hyperparameters.
    X_train, y_train : training data
    X_val, y_val : validation data

    Returns
    -------
    float
        F1 score on the validation set.
    """
    # Define search space
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "n_jobs": -1,
        "verbosity": 0,
    }

    # Train model
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Predict on validation set
    y_pred = model.predict(X_val)

    # Calculate F1 score
    f1 = f1_score(y_val, y_pred)

    return f1


def optuna_tune_xgboost(X_train, y_train, X_val, y_val, n_trials=15, output_dir="tuning/"):
    """
    Runs Optuna hyperparameter tuning for XGBoost.

    Parameters
    ----------
    X_train, y_train : training data
    X_val, y_val : validation data
    n_trials : int
        Number of Optuna trials (default=15).
    output_dir : str
        Directory to save best parameters.

    Returns
    -------
    dict
        Best parameters found by Optuna.
    """
    os.makedirs(output_dir, exist_ok=True)

    def wrapped_objective(trial):
        return objective(trial, X_train, y_train, X_val, y_val)

    study = optuna.create_study(direction="maximize")
    study.optimize(wrapped_objective, n_trials=n_trials)

    best_trial = study.best_trial
    best_params = best_trial.params

    print("\nBest Trial:")
    print(f"  Value (F1): {best_trial.value}")
    print("  Params:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")

    # Save best parameters
    best_params_path = os.path.join(output_dir, "xgboost_optuna.json")
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=4)

    # Save study for future reference
    joblib.dump(study, os.path.join(output_dir, "optuna_study.pkl"))

    return best_params


if __name__ == "__main__":
    # Example usage with sample pre-split data
    print("Loading training and validation data...")
    X_train = pd.read_csv("data/splits/X_train.csv")
    y_train = pd.read_csv("data/splits/y_train.csv").squeeze()
    X_val = pd.read_csv("data/splits/X_val.csv")
    y_val = pd.read_csv("data/splits/y_val.csv").squeeze()

    print("Running Optuna hyperparameter tuning for XGBoost...")
    best_params = optuna_tune_xgboost(X_train, y_train, X_val, y_val, n_trials=15)

    print("\nBest parameters found:")
    print(best_params)
