"""
Visualizations Module
====================

Provides visualization capabilities for the fraud detection system.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any

plt.style.use('default')
sns.set_palette("husl")


class TransactionVisualizer:
    """Handles visualizations for transaction analysis."""
    
    def __init__(self):
        self.figsize = (10, 6)
    
    def plot_transaction_distribution(self, df: pd.DataFrame):
        """Plot distribution of legitimate vs suspicious transactions."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if 'label' in df.columns:
            counts = df['label'].value_counts()
            labels = ['Legitimate', 'Suspicious']
            ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], 
                   color=['lightblue', 'salmon'])
            ax.set_title('Transaction Distribution')
            ax.set_ylabel('Count')
        else:
            ax.text(0.5, 0.5, 'No label column found', ha='center', va='center')
            ax.set_title('Transaction Distribution - No Labels')
        
        return fig
    
    def plot_amount_distribution(self, df: pd.DataFrame):
        """Plot distribution of transaction amounts."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if 'amount' in df.columns:
            ax.hist(df['amount'], bins=30, alpha=0.7, color='steelblue')
            ax.set_title('Transaction Amount Distribution')
            ax.set_xlabel('Amount')
            ax.set_ylabel('Frequency')
        else:
            ax.text(0.5, 0.5, 'No amount column found', ha='center', va='center')
            ax.set_title('Amount Distribution - No Data')
        
        return fig
    
    def plot_correlation_heatmap(self, df: pd.DataFrame):
        """Plot correlation heatmap of numerical features."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Select only numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Feature Correlation Heatmap')
        else:
            ax.text(0.5, 0.5, 'Insufficient numerical columns for correlation', 
                   ha='center', va='center')
            ax.set_title('Correlation Heatmap - Insufficient Data')
        
        return fig
    
    def plot_model_comparison(self, performance_metrics: list):
        """Plot model performance comparison."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if performance_metrics:
            df_metrics = pd.DataFrame(performance_metrics)
            if 'Test Score' in df_metrics.columns:
                ax.bar(df_metrics['Model'], df_metrics['Test Score'], color='lightgreen')
                ax.set_title('Model Performance Comparison')
                ax.set_ylabel('Test Score')
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.text(0.5, 0.5, 'No test scores available', ha='center', va='center')
        else:
            ax.text(0.5, 0.5, 'No performance metrics available', ha='center', va='center')
        
        return fig
    
    def plot_feature_importance(self, importance_dict: Dict[str, float], model_name: str):
        """Plot feature importance for a model."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if importance_dict:
            features = list(importance_dict.keys())
            importances = list(importance_dict.values())
            
            # Sort by importance
            sorted_idx = np.argsort(importances)
            features_sorted = [features[i] for i in sorted_idx]
            importances_sorted = [importances[i] for i in sorted_idx]
            
            ax.barh(features_sorted, importances_sorted, color='lightcoral')
            ax.set_title(f'Feature Importance - {model_name}')
            ax.set_xlabel('Importance')
        else:
            ax.text(0.5, 0.5, 'No feature importance data', ha='center', va='center')
        
        return fig
    
    def plot_transaction_by_country(self, df: pd.DataFrame):
        """Plot transactions by country."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if 'sender_country' in df.columns:
            country_counts = df['sender_country'].value_counts().head(10)
            ax.bar(country_counts.index, country_counts.values, color='skyblue')
            ax.set_title('Top 10 Countries by Transaction Count')
            ax.set_xlabel('Country')
            ax.set_ylabel('Transaction Count')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No country data available', ha='center', va='center')
        
        return fig
    
    def plot_cross_border_analysis(self, df: pd.DataFrame):
        """Plot cross-border transaction analysis."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if 'sender_country' in df.columns and 'receiver_country' in df.columns:
            cross_border = (df['sender_country'] != df['receiver_country']).value_counts()
            labels = ['Domestic', 'Cross-border']
            ax.pie(cross_border.values, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.set_title('Domestic vs Cross-border Transactions')
        else:
            ax.text(0.5, 0.5, 'No country data for cross-border analysis', ha='center', va='center')
        
        return fig
    
    def plot_suspicious_by_country(self, df: pd.DataFrame):
        """Plot suspicious transactions by country."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if 'sender_country' in df.columns and 'label' in df.columns:
            country_suspicious = df.groupby('sender_country')['label'].agg(['count', 'sum'])
            country_suspicious['rate'] = country_suspicious['sum'] / country_suspicious['count']
            top_countries = country_suspicious.nlargest(10, 'count')
            
            ax.bar(top_countries.index, top_countries['rate'], color='orange')
            ax.set_title('Suspicious Transaction Rate by Country')
            ax.set_xlabel('Country')
            ax.set_ylabel('Suspicious Rate')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No data for country analysis', ha='center', va='center')
        
        return fig
    
    def plot_amount_vs_suspicious(self, df: pd.DataFrame):
        """Plot amount distribution for suspicious vs legitimate transactions."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if 'amount' in df.columns and 'label' in df.columns:
            legitimate = df[df['label'] == 0]['amount']
            suspicious = df[df['label'] == 1]['amount']
            
            ax.hist(legitimate, bins=30, alpha=0.7, label='Legitimate', color='lightblue')
            ax.hist(suspicious, bins=30, alpha=0.7, label='Suspicious', color='salmon')
            ax.set_title('Amount Distribution: Legitimate vs Suspicious')
            ax.set_xlabel('Amount')
            ax.set_ylabel('Frequency')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No amount or label data', ha='center', va='center')
        
        return fig
    
    def plot_hourly_patterns(self, df: pd.DataFrame):
        """Plot transaction patterns by hour."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if 'hour' in df.columns:
            hourly_counts = df['hour'].value_counts().sort_index()
            ax.plot(hourly_counts.index, hourly_counts.values, marker='o', color='teal')
            ax.set_title('Transaction Patterns by Hour')
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Transaction Count')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No hourly data available', ha='center', va='center')
        
        return fig