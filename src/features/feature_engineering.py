# src/features/feature_engineering.py
"""
Feature engineering utilities for transaction fraud detection.

Includes:
- Basic engineered features (time, log amount, geo mismatch).
- Advanced behavior-based features (velocity, entropy, rolling aggregates).
- A unified pipeline that can be extended with advanced features.

Author: Senior ML Engineer & Fraud Detection Expert
"""

from typing import Tuple, List
import pandas as pd
import numpy as np
import logging

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(ch)


# ---------------------------------------------------------------------------
# BASIC FEATURES
# ---------------------------------------------------------------------------
def extract_time_features(df, ts_col='timestamp'):
    """Extract time-based features from timestamp column"""
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df['txn_hour'] = df[ts_col].dt.hour
    df['txn_day'] = df[ts_col].dt.day
    df['txn_weekday'] = df[ts_col].dt.weekday
    return df


def add_log_amount(df, col='amount'):
    """Add logarithmic transformation of amount"""
    df = df.copy()
    df['txn_amount_log'] = np.log1p(df[col])
    return df


def add_geo_mismatch(df):
    """Add flag for geographical mismatch"""
    df = df.copy()
    df['geo_mismatch_flag'] = (df['sender_country'] != df['receiver_country']).astype(int)
    return df

def get_features(df):
    """
    Main function to get all engineered features
    Returns: (X_features, feature_list)
    """
    # Apply all feature engineering steps
    df_engineered = extract_time_features(df)
    df_engineered = add_log_amount(df_engineered)
    df_engineered = add_geo_mismatch(df_engineered)
    
    # Select only the numerical features for modeling
    feature_columns = ['txn_hour', 'txn_day', 'txn_weekday', 'txn_amount_log', 'geo_mismatch_flag']
    X_features = df_engineered[feature_columns].copy()
    
    return X_features, feature_columns

# ---------------------------------------------------------------------------
# ADVANCED FEATURES
# ---------------------------------------------------------------------------
def add_sender_avg_txn_amount_last_30d(df: pd.DataFrame) -> pd.DataFrame:
    if not {"sender_id", "timestamp", "amount"}.issubset(df.columns):
        raise ValueError("Missing required columns for avg txn amount feature.")

    df_out = df.copy()
    df_out["timestamp"] = pd.to_datetime(df_out["timestamp"])
    df_out = df_out.sort_values(["sender_id", "timestamp"])
    df_out["amount_numeric"] = pd.to_numeric(df_out["amount"], errors="coerce")

    df_idx = df_out.set_index("timestamp")
    sender_avg = (
        df_idx.groupby("sender_id")["amount_numeric"]
        .rolling("30D", min_periods=1)
        .mean()
        .reset_index(name="sender_avg_txn_amount_last_30d")
    )
    return df_out.merge(sender_avg, on=["sender_id", "timestamp"], how="left")


def add_sender_txn_velocity(df: pd.DataFrame) -> pd.DataFrame:
    if not {"sender_id", "timestamp"}.issubset(df.columns):
        raise ValueError("Missing required columns for txn velocity feature.")

    df_out = df.copy()
    df_out["timestamp"] = pd.to_datetime(df_out["timestamp"])
    df_out = df_out.sort_values(["sender_id", "timestamp"])

    df_out["sender_txn_velocity_1h"] = (
        df_out.groupby("sender_id")
        .timestamp.rolling("1H")
        .count()
        .reset_index(level=0, drop=True)
    ).astype(int)

    return df_out


def add_unique_receivers_last_7d(df: pd.DataFrame) -> pd.DataFrame:
    if not {"sender_id", "receiver_id", "timestamp"}.issubset(df.columns):
        raise ValueError("Missing required columns for unique receivers feature.")

    df_out = df.copy()
    df_out["timestamp"] = pd.to_datetime(df_out["timestamp"])
    df_sorted = df_out.sort_values(["sender_id", "timestamp"]).set_index("timestamp")

    uniq_recv = (
        df_sorted.groupby("sender_id")["receiver_id"]
        .rolling("7D")
        .apply(lambda x: x.nunique(), raw=False)
        .reset_index(name="unique_receivers_last_7d")
    )
    return df_out.merge(uniq_recv, on=["sender_id", "timestamp"], how="left")


def add_receiver_entropy_of_senders(df: pd.DataFrame) -> pd.DataFrame:
    if not {"receiver_id", "sender_id", "timestamp"}.issubset(df.columns):
        raise ValueError("Missing required columns for entropy feature.")

    def entropy(series):
        probs = series.value_counts(normalize=True)
        return -(probs * np.log2(probs + 1e-9)).sum()

    df_out = df.copy()
    df_out["timestamp"] = pd.to_datetime(df_out["timestamp"])
    df_sorted = df_out.sort_values("timestamp").set_index("timestamp")

    rec_entropy = (
        df_sorted.groupby("receiver_id")["sender_id"]
        .rolling("30D")
        .apply(lambda x: entropy(x), raw=False)
        .reset_index(name="receiver_entropy_of_senders")
    )
    return df_out.merge(rec_entropy, on=["receiver_id", "timestamp"], how="left")


def add_avg_inter_txn_time_sender(df: pd.DataFrame) -> pd.DataFrame:
    if not {"sender_id", "timestamp"}.issubset(df.columns):
        raise ValueError("Missing required columns for inter-txn time feature.")

    df_out = df.copy()
    df_out["timestamp"] = pd.to_datetime(df_out["timestamp"])
    df_out = df_out.sort_values(["sender_id", "timestamp"])

    df_out["prev_ts"] = df_out.groupby("sender_id")["timestamp"].shift(1)
    df_out["inter_txn_seconds"] = (df_out["timestamp"] - df_out["prev_ts"]).dt.total_seconds()
    df_out["avg_inter_txn_time_sender"] = (
        df_out.groupby("sender_id")["inter_txn_seconds"]
        .transform(lambda s: s.rolling(window=5, min_periods=1).mean())
    )
    df_out = df_out.drop(columns=["prev_ts"])
    return df_out


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        from src.data_collection import get_sample_df
    except Exception as e:
        logger.error("Could not import sample data: %s", e)
        raise

    raw_df = get_sample_df()
    logger.info("Original DataFrame shape: %s", raw_df.shape)

    X_basic, feats = get_features(raw_df)
    print("\n=== Basic Features Preview ===")
    print(X_basic.head())

    # Example advanced feature demo
    df_adv = add_sender_avg_txn_amount_last_30d(raw_df)
    print("\n=== Advanced Feature: sender_avg_txn_amount_last_30d ===")
    print(df_adv[["sender_id", "timestamp", "sender_avg_txn_amount_last_30d"]].head())
