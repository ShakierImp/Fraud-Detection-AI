"""
Feature Engineering Module
==========================

Provides feature engineering capabilities for the fraud detection system.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature engineering for fraud detection."""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def create_features(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Create engineered features from transaction data.
        
        Args:
            df: Input DataFrame
            **kwargs: Feature creation options
            
        Returns:
            DataFrame with engineered features
        """
        df_features = df.copy()
        
        # Time-based features
        if 'timestamp' in df_features.columns:
            df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
            df_features['hour'] = df_features['timestamp'].dt.hour
            df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
            df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6])
        
        # Amount-based features
        if 'amount' in df_features.columns:
            df_features['amount_log'] = np.log1p(df_features['amount'])
            df_features['amount_zscore'] = (df_features['amount'] - df_features['amount'].mean()) / df_features['amount'].std()
        
        # Geographic features
        if 'sender_country' in df_features.columns and 'receiver_country' in df_features.columns:
            df_features['cross_border'] = (df_features['sender_country'] != df_features['receiver_country']).astype(int)
        
        return df_features
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numerical features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with normalized features
        """
        df_norm = df.copy()
        
        # Select numerical columns
        numerical_cols = df_norm.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            df_norm[numerical_cols] = self.scaler.fit_transform(df_norm[numerical_cols])
        
        return df_norm