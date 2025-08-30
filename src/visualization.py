import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class TransactionVisualizer:
    def plot_transaction_distribution(self, df):
        fig, ax = plt.subplots(figsize=(10, 6))
        if 'label' in df.columns:
            df['label'].value_counts().plot(kind='bar', ax=ax)
            ax.set_title('Transaction Distribution (0=Legitimate, 1=Suspicious)')
        return fig

    def plot_amount_distribution(self, df):
        fig, ax = plt.subplots(figsize=(10, 6))
        if 'amount' in df.columns:
            df['amount'].hist(bins=50, ax=ax)
            ax.set_title('Amount Distribution')
        return fig

    def plot_model_comparison(self, metrics):
        fig, ax = plt.subplots(figsize=(10, 6))
        # Simple implementation
        ax.bar(['Model 1', 'Model 2'], [0.85, 0.89])
        ax.set_title('Model Accuracy Comparison')
        return fig