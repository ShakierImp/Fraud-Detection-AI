import pandas as pd
import numpy as np

class DataProcessor:
    def load_and_validate(self, df):
        # Basic data validation and cleaning
        df = df.dropna()
        df = df.drop_duplicates()
        
        # Ensure required columns
        required_cols = ['amount', 'sender_country', 'receiver_country']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Add label column if not present
        if 'label' not in df.columns:
            df['label'] = 0  # Default to legitimate
            
        return df