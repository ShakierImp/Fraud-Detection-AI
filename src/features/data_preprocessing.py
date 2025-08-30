"""
Data Preprocessing Module
=========================

This module provides robust and immutable preprocessing functions
for cleaning, transforming, and saving transaction data. 

Core principles:
- **Immutability**: Each function returns a new DataFrame copy 
  without modifying the input.
- **Robustness**: Handles missing values, duplicates, and outliers gracefully.
- **Logging**: All key steps are logged for better traceability.

Author: Senior Data Engineer
"""

import pandas as pd
import numpy as np
import logging
from typing import List

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the input DataFrame.

    This function performs data cleaning steps on a copy of the input DataFrame:
        1. Drops duplicate rows (keeping the first occurrence).
        2. Converts the 'timestamp' column to datetime format.
        3. Ensures the 'amount' column is numeric (coercing errors to NaN).
        4. Creates a new boolean feature 'is_weekend' based on transaction day.
        5. Handles missing values:
            - Numerical columns: fill NaN with column median.
            - Categorical columns: fill NaN with 'UNKNOWN'.

    Args:
        df (pd.DataFrame): Input pandas DataFrame with transaction data.

    Returns:
        pd.DataFrame: Cleaned DataFrame (new copy).

    Raises:
        ValueError: If required columns ('timestamp', 'amount') are missing.
    """
    df_cleaned = df.copy()

    # Check required columns
    required_cols = ["timestamp", "amount"]
    for col in required_cols:
        if col not in df_cleaned.columns:
            raise ValueError(f"Missing required column: {col}")

    # Drop duplicates
    before = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates(keep="first")
    after = len(df_cleaned)
    logger.info("Dropped %d duplicate rows", before - after)

    # Convert timestamp to datetime
    df_cleaned["timestamp"] = pd.to_datetime(
        df_cleaned["timestamp"], errors="coerce"
    )
    logger.info("Converted 'timestamp' to datetime")

    # Ensure 'amount' is numeric
    df_cleaned["amount"] = pd.to_numeric(
        df_cleaned["amount"], errors="coerce"
    )
    logger.info("Converted 'amount' to numeric")

    # Create 'is_weekend' feature
    df_cleaned["is_weekend"] = df_cleaned["timestamp"].dt.dayofweek >= 5
    logger.info("Created 'is_weekend' feature")

    # Handle missing values
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype in [np.float64, np.int64]:
            median_val = df_cleaned[col].median()
            n_missing = df_cleaned[col].isna().sum()
            df_cleaned[col] = df_cleaned[col].fillna(median_val)
            if n_missing > 0:
                logger.info("Filled %d NaN values in '%s' with median", n_missing, col)
        else:
            n_missing = df_cleaned[col].isna().sum()
            df_cleaned[col] = df_cleaned[col].fillna("UNKNOWN")
            if n_missing > 0:
                logger.info("Filled %d NaN values in '%s' with 'UNKNOWN'", n_missing, col)

    return df_cleaned


def handle_outliers(
    df: pd.DataFrame, col: str = "amount", method: str = "iqr", k: float = 1.5
) -> pd.DataFrame:
    """Handle outliers in a numerical column by capping values.

    By default, uses the Interquartile Range (IQR) method to cap outliers.

    Args:
        df (pd.DataFrame): Input pandas DataFrame.
        col (str): Column to handle outliers for. Default is 'amount'.
        method (str): Outlier detection method. Currently supports 'iqr'. Default 'iqr'.
        k (float): Multiplier for IQR range. Default is 1.5.

    Returns:
        pd.DataFrame: DataFrame with outliers capped (new copy).

    Raises:
        ValueError: If the specified column is missing or method is unsupported.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")

    if method != "iqr":
        raise ValueError(f"Unsupported method '{method}'. Only 'iqr' is supported.")

    df_out = df.copy()

    Q1 = df_out[col].quantile(0.25)
    Q3 = df_out[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR

    before_min = (df_out[col] < lower_bound).sum()
    before_max = (df_out[col] > upper_bound).sum()

    df_out[col] = np.clip(df_out[col], lower_bound, upper_bound)

    logger.info(
        "Outliers capped in '%s': %d below lower bound, %d above upper bound",
        col,
        before_min,
        before_max,
    )

    return df_out


def drop_unused_cols(df: pd.DataFrame, cols_to_drop: List[str]) -> pd.DataFrame:
    """Drop unused columns from the DataFrame.

    Args:
        df (pd.DataFrame): Input pandas DataFrame.
        cols_to_drop (List[str]): List of column names to drop.

    Returns:
        pd.DataFrame: DataFrame with specified columns dropped (new copy).
    """
    df_new = df.copy()
    for col in cols_to_drop:
        if col in df_new.columns:
            df_new = df_new.drop(columns=[col])
            logger.info("Dropped column: %s", col)
        else:
            logger.warning("Column '%s' does not exist and cannot be dropped", col)
    return df_new


def save_processed(df: pd.DataFrame, path: str) -> None:
    """Save processed DataFrame to CSV.

    Args:
        df (pd.DataFrame): DataFrame to save.
        path (str): Destination file path.
    """
    df.to_csv(path, index=False)
    logger.info("Processed DataFrame saved to %s", path)


# ---------------------------------------------------------------------------
# CLI DEMO
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_collection import get_sample_df

    logger.info("Starting data preprocessing demo...")

    # Load sample data
    df = get_sample_df()

    # Intentionally add a duplicate row for testing
    df_with_dupes = pd.concat([df, df.head(1)], ignore_index=True)

    # Clean the DataFrame
    cleaned_df = clean_df(df_with_dupes)

    print(f"Original df length: {len(df)}")
    print(f"DF with duplicates length: {len(df_with_dupes)}")
    print(
        f"After cleaning length: {len(cleaned_df)}. "
        f"Duplicates removed: {len(df_with_dupes) - len(cleaned_df)}"
    )

    # Handle outliers
    df_no_outliers = handle_outliers(cleaned_df, col="amount")

    # Drop unused columns (example: dropping device_id if not needed)
    df_dropped = drop_unused_cols(df_no_outliers, ["device_id", "non_existent_col"])

    # Save processed data
    save_processed(df_dropped, "processed_transactions.csv")

    logger.info("Demo complete.")
