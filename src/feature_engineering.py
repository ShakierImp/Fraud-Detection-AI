import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class FeatureEngineer:
    def create_features(self, df, time_features=True, amount_features=True, 
                       country_features=True, device_features=True, user_features=True):
        
        df = df.copy()
        
        # Time-based features
        if time_features and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Amount-based features
        if amount_features and 'amount' in df.columns:
            df['amount_log'] = np.log1p(df['amount'])
            df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
        
        # Country-based features
        if country_features and all(col in df.columns for col in ['sender_country', 'receiver_country']):
            df['is_cross_border'] = (df['sender_country'] != df['receiver_country']).astype(int)
            
            # Country risk scoring
            high_risk_countries = ['RUS', 'IRN', 'PRK', 'SYR']
            df['sender_risk'] = df['sender_country'].apply(lambda x: 1 if x in high_risk_countries else 0)
            df['receiver_risk'] = df['receiver_country'].apply(lambda x: 1 if x in high_risk_countries else 0)
        
        return df

    def normalize_features(self, df):
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'label' in numerical_cols:
            numerical_cols.remove('label')
        
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        return df