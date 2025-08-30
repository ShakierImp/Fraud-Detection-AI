from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd

class SuspiciousTransactionModels:
    def prepare_data(self, df):
        # Select numerical columns and label
        X = df.select_dtypes(include=[np.number]).drop('label', axis=1, errors='ignore')
        y = df['label'] if 'label' in df.columns else pd.Series([0] * len(df))
        
        # Handle missing values
        X = X.fillna(0)
        
        return X, y

    def train_models(self, X, y, models=None, test_size=0.2, 
                    handle_imbalance="None", cross_validation=True):
        
        if models is None:
            models = ["Random Forest", "XGBoost"]
        
        # Handle class imbalance
        if handle_imbalance == "SMOTE" and len(np.unique(y)) > 1:
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        results = {
            'performance_metrics': [],
            'trained_models': {},
            'feature_importance': {}
        }
        
        # Train selected models
        for model_name in models:
            if model_name == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_name == "XGBoost":
                model = XGBClassifier(random_state=42)
            elif model_name == "Logistic Regression":
                model = LogisticRegression(random_state=42)
            elif model_name == "SVM":
                model = SVC(probability=True, random_state=42)
            else:
                continue
            
            model.fit(X_train, y_train)
            results['trained_models'][model_name] = {'model': model}
        
        return results