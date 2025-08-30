# src/models/hyperparameter_tuning.py

import os
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def grid_search_rf(X_train, y_train, param_grid=None, cv=3, scoring='f1'):
    """
    Perform hyperparameter tuning for a Random Forest classifier using GridSearchCV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    param_grid : dict, optional
        Grid of parameters to search. If None, uses a small demo grid.
    cv : int, default=3
        Number of cross-validation folds.
    scoring : str, default='f1'
        Scoring metric for evaluation.

    Returns
    -------
    GridSearchCV
        The fitted GridSearchCV object containing best estimator and results.
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'class_weight': ['balanced']
        }

    rf_clf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf_clf,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    return grid_search


def save_best_params(grid_search_result, output_path='tuning/best_params_rf.json'):
    """
    Save the best parameters from a GridSearchCV result to a JSON file.

    Parameters
    ----------
    grid_search_result : GridSearchCV
        The fitted GridSearchCV object.
    output_path : str, default='tuning/best_params_rf.json'
        Path to save the best parameters JSON.
    """
    best_params = grid_search_result.best_params_
    best_score = grid_search_result.best_score_

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(best_params, f, indent=4)

    print(f"Best Parameters: {best_params}")
    print(f"Best F1 Score: {best_score:.4f}")
    print(f"Saved best parameters to {output_path}")


if __name__ == '__main__':
    # Demo workflow
    print("Loading training data...")
    X_train = pd.read_csv('data/splits/X_train.csv')
    y_train = pd.read_csv('data/splits/y_train.csv').squeeze()

    print("Starting hyperparameter tuning...")
    grid_result = grid_search_rf(X_train, y_train)

    print("Saving best parameters...")
    save_best_params(grid_result)

    print("Hyperparameter tuning complete.")
