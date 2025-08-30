"""
Data Processor Module
====================

Handles data loading, validation, and basic preprocessing for the fraud detection system.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data loading and validation for fraud detection."""
    
    def __init__(self):
        self.required_columns = [
            'transaction_id', 'timestamp', 'amount', 'sender_id', 
            'receiver_id', 'sender_country', 'receiver_country', 'device_id'
        ]
    
    def load_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Load and validate transaction data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Validated and processed DataFrame
        """
        # Check for required columns
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
        
        # Add label column if missing (for demo purposes)
        if 'label' not in df.columns:
            # Generate synthetic labels for demo
            np.random.seed(42)
            df['label'] = np.random.choice([0, 1], size=len(df), p=[0.95, 0.05])
        
        # Basic data type conversions
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        return df