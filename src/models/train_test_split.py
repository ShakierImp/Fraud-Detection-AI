"""
train_test_split.py

Module for performing reproducible stratified train/validation/test splits
and saving them for machine learning workflows.

Author: Senior ML Engineer
"""

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Perform a two-step stratified split to create train, validation, and test sets.

    Step 1: Split the dataset into train_val and test sets.
    Step 2: Split train_val further into train and validation sets.

    Args:
        X (array-like or pd.DataFrame): Feature matrix.
        y (array-like or pd.Series): Target labels.
        test_size (float, optional): Proportion of the dataset for the test split. Default is 0.2.
        val_size (float, optional): Proportion of the dataset for the validation split. 
                                    This is relative to the full dataset (not train_val). Default is 0.1.
        random_state (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
            - X_train (pd.DataFrame)
            - X_val (pd.DataFrame)
            - X_test (pd.DataFrame)
            - y_train (pd.Series)
            - y_val (pd.Series)
            - y_test (pd.Series)
    """
    # Step 1: Split into train_val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Step 2: Split train_val into train and validation
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_size, stratify=y_train_val, random_state=random_state
    )

    # Convert to DataFrame/Series if not already
    X_train = pd.DataFrame(X_train)
    X_val = pd.DataFrame(X_val)
    X_test = pd.DataFrame(X_test)

    y_train = pd.Series(y_train).reset_index(drop=True)
    y_val = pd.Series(y_val).reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True)

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_splits(X_train, X_val, X_test, y_train, y_val, y_test, out_dir='data/splits/'):
    """
    Save train/validation/test splits to disk as CSV files, along with class distributions.

    Args:
        X_train (pd.DataFrame): Training features.
        X_val (pd.DataFrame): Validation features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training labels.
        y_val (pd.Series): Validation labels.
        y_test (pd.Series): Test labels.
        out_dir (str, optional): Output directory for saving splits. Default is 'data/splits/'.

    Returns:
        None
    """
    os.makedirs(out_dir, exist_ok=True)

    # Save feature splits
    X_train.to_csv(os.path.join(out_dir, "X_train.csv"), index=False)
    X_val.to_csv(os.path.join(out_dir, "X_val.csv"), index=False)
    X_test.to_csv(os.path.join(out_dir, "X_test.csv"), index=False)

    # Save target splits
    y_train.to_csv(os.path.join(out_dir, "y_train.csv"), index=False, header=True)
    y_val.to_csv(os.path.join(out_dir, "y_val.csv"), index=False, header=True)
    y_test.to_csv(os.path.join(out_dir, "y_test.csv"), index=False, header=True)

    # Save class distribution
    distribution = {
        "train": y_train.value_counts().to_dict(),
        "val": y_val.value_counts().to_dict(),
        "test": y_test.value_counts().to_dict()
    }

    with open(os.path.join(out_dir, "distribution.json"), "w") as f:
        json.dump(distribution, f, indent=2)


if __name__ == "__main__":
    # Demo: Generate sample data and perform split
    from sklearn.datasets import make_classification

    # Generate imbalanced classification dataset
    X, y = make_classification(
        n_samples=1000, n_features=5, n_informative=5, n_redundant=0,
        random_state=42, weights=[0.9]
    )

    # Perform stratified split
    splits = stratified_split(X, y, test_size=0.2, val_size=0.1, random_state=42)

    # Save splits
    save_splits(*splits, out_dir='data/splits/')

    # Print distribution
    with open('data/splits/distribution.json', 'r') as f:
        dist = json.load(f)

    print("Class distribution per split:")
    print(json.dumps(dist, indent=2))
