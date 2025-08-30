"""
Models Module
=============

Provides model training and evaluation capabilities for fraud detection.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)


class SuspiciousTransactionModels:
    """Handles model training and evaluation for fraud detection."""
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42)
        }
        self.trained_models = {}
    
    def prepare_data(self, df: pd.DataFrame):
        """
        Prepare features and target from DataFrame.
        
        Args:
            df: Input DataFrame with features and label
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        # Select numerical features for modeling
        feature_cols = [col for col in df.columns if col not in ['label', 'transaction_id', 'timestamp']]
        numerical_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        
        X = df[numerical_cols].fillna(0)
        y = df['label'] if 'label' in df.columns else pd.Series([0] * len(df))
        
        return X, y
    
    def train_models(self, X, y, models=None, test_size=0.2, handle_imbalance='None', cross_validation=True):
        """
        Train multiple models on the data.
        
        Args:
            X: Feature matrix
            y: Target vector
            models: List of model names to train
            test_size: Proportion for test split
            handle_imbalance: Method to handle class imbalance
            cross_validation: Whether to use cross validation
            
        Returns:
            Dictionary with training results
        """
        if models is None:
            models = ['Random Forest', 'XGBoost']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Handle class imbalance
        if handle_imbalance == 'SMOTE':
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        elif handle_imbalance == 'Random Undersampling':
            undersampler = RandomUnderSampler(random_state=42)
            X_train, y_train = undersampler.fit_resample(X_train, y_train)
        
        results = {
            'trained_models': {},
            'performance_metrics': [],
            'feature_importance': {}
        }
        
        for model_name in models:
            if model_name not in self.models:
                continue
                
            logger.info(f"Training {model_name}...")
            model = self.models[model_name]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Cross validation if requested
            cv_scores = None
            if cross_validation:
                cv_scores = cross_val_score(model, X_train, y_train, cv=3)
            
            # Store results
            self.trained_models[model_name] = {
                'model': model,
                'feature_columns': list(X.columns)
            }
            
            results['trained_models'][model_name] = {
                'model': model,
                'feature_columns': list(X.columns)
            }
            
            results['performance_metrics'].append({
                'Model': model_name,
                'Train Score': train_score,
                'Test Score': test_score,
                'CV Mean': cv_scores.mean() if cv_scores is not None else None,
                'CV Std': cv_scores.std() if cv_scores is not None else None
            })
            
            # Feature importance if available
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(X.columns, model.feature_importances_))
                results['feature_importance'][model_name] = importance_dict
        
        return results