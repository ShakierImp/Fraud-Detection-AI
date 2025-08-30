"""
tests/test_preprocessing.py

Pytest test suite for validating data preprocessing and feature engineering
functions in the FraudGuardian AI project.

Functions under test:
- clean_df: from src.features.data_preprocessing
- get_features: from src.features.feature_engineering

This suite ensures data cleaning, immutability, and feature correctness.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.features.data_preprocessing import clean_df
from src.features.feature_engineering import get_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_df():
    """
    Fixture that loads the sample input CSV once per test module.
    If the file is missing, skip the tests gracefully.
    """
    data_path = os.path.join("data", "sample_input.csv")
    if not os.path.exists(data_path):
        pytest.skip("Sample input CSV not found. Skipping preprocessing tests.")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        pytest.skip(f"Could not load sample input CSV: {e}")
    return df


# ---------------------------------------------------------------------------
# Tests for clean_df
# ---------------------------------------------------------------------------

def test_clean_df_removes_duplicates(sample_df):
    """
    clean_df should remove duplicate rows.
    """
    duplicated_df = pd.concat([sample_df, sample_df], ignore_index=True)
    cleaned_df = clean_df(duplicated_df)

    assert len(cleaned_df) == len(sample_df), \
        "clean_df did not remove duplicates correctly."


def test_clean_df_returns_copy(sample_df):
    """
    clean_df should return a new DataFrame, not modify the original.
    """
    cleaned_df = clean_df(sample_df)
    assert cleaned_df is not sample_df, \
        "clean_df should return a new object, not the same DataFrame."


def test_clean_df_handles_missing_values(sample_df):
    """
    clean_df should handle missing values by imputing or removing them.
    """
    df_with_nans = sample_df.copy()
    # Introduce NaNs in both numerical and categorical columns
    if not df_with_nans.empty:
        first_col = df_with_nans.columns[0]
        df_with_nans.loc[0, first_col] = np.nan
        if len(df_with_nans.columns) > 1:
            second_col = df_with_nans.columns[1]
            df_with_nans.loc[1, second_col] = np.nan

    cleaned_df = clean_df(df_with_nans)
    assert not cleaned_df.isna().any().any(), \
        "clean_df should handle missing values, but NaNs remain."


# ---------------------------------------------------------------------------
# Tests for get_features
# ---------------------------------------------------------------------------

def test_get_features_returns_expected_columns(sample_df):
    """
    get_features should return a DataFrame with the exact expected feature columns.
    """
    cleaned_df = clean_df(sample_df)
    X_features, feature_list = get_features(cleaned_df)

    expected_columns = [
        "txn_hour",
        "txn_day",
        "txn_weekday",
        "txn_amount_log",
        "geo_mismatch_flag",
    ]

    assert list(X_features.columns) == expected_columns, \
        f"Expected columns {expected_columns}, but got {list(X_features.columns)}."
    assert isinstance(feature_list, list), \
        "feature_list should be a list."
    assert feature_list == expected_columns, \
        "feature_list does not match the expected columns."


def test_get_features_returns_numeric_only(sample_df):
    """
    get_features should return only numeric columns.
    """
    cleaned_df = clean_df(sample_df)
    X_features, _ = get_features(cleaned_df)

    non_numeric = [col for col in X_features.columns
                   if not np.issubdtype(X_features[col].dtype, np.number)]

    assert not non_numeric, \
        f"Found non-numeric columns in features: {non_numeric}"


def test_get_features_immutable(sample_df):
    """
    get_features should not modify the input DataFrame.
    """
    input_copy = sample_df.copy(deep=True)
    _ = get_features(sample_df)

    pd.testing.assert_frame_equal(sample_df, input_copy,
                                  check_dtype=False,
                                  obj="Input DataFrame after get_features")
