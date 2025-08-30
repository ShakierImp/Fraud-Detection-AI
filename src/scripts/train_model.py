#!/usr/bin/env python3
"""
Training Script for FraudGuardian AI
====================================

This script provides a complete training pipeline for fraud detection models.
It handles data loading, preprocessing, feature engineering, model training,
and model persistence.

Usage:
    python src/scripts/train_model.py [--fast]
    
Options:
    --fast: Use a smaller dataset and fewer estimators for quick training
"""

import argparse
import os
import sys
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data_collection import get_sample_df
from src.features.data_preprocessing import clean_df
from src.features.feature_engineering import get_features
from src.models.model_serializer import save_model_artifacts
from src.models.train_test_split import stratified_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(fast_mode=False):
    """
    Main training function.
    
    Args:
        fast_mode (bool): If True, use reduced parameters for quick training
    """
    logger.info("Starting model training pipeline...")
    
    # Load data
    logger.info("Loading sample dataset...")
    df = get_sample_df()
    logger.info(f"Loaded dataset with shape: {df.shape}")
    
    # Clean data
    logger.info("Cleaning data...")
    df_clean = clean_df(df)
    logger.info(f"Cleaned dataset shape: {df_clean.shape}")
    
    # Feature engineering
    logger.info("Extracting features...")
    X_features, feature_list = get_features(df_clean)
    
    # Get target variable
    if 'label' in df_clean.columns:
        y = df_clean['label']
    else:
        # Generate synthetic labels for demo
        import numpy as np
        np.random.seed(42)
        y = np.random.choice([0, 1], size=len(df_clean), p=[0.95, 0.05])
    
    logger.info(f"Features shape: {X_features.shape}, Target shape: {y.shape}")
    
    # Train-test split
    logger.info("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        X_features, y, test_size=0.2, val_size=0.1, random_state=42
    )
    
    # Model parameters
    if fast_mode:
        params = {
            'n_estimators': 10,
            'max_depth': 5,
            'random_state': 42,
            'class_weight': 'balanced',
            'n_jobs': -1
        }
        logger.info("Using fast mode parameters")
    else:
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'class_weight': 'balanced',
            'n_jobs': -1
        }
        logger.info("Using full training parameters")
    
    # Train model
    logger.info("Training Random Forest model...")
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Evaluate
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"Model F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model artifacts
    logger.info("Saving model artifacts...")
    metadata = save_model_artifacts(
        model=model,
        feature_list=feature_list,
        model_name='random_forest',
        scaler=None
    )
    
    logger.info(f"Model saved with metadata: {metadata}")
    logger.info("Training pipeline completed successfully!")
    
    return model, metadata


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--fast", action="store_true", help="Use fast training mode")
    
    args = parser.parse_args()
    
    try:
        model, metadata = train_model(fast_mode=args.fast)
        print(f"\n✅ Training completed successfully!")
        print(f"Model version: {metadata['version']}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()