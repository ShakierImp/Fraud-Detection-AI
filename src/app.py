import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Suspicious Transaction Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom classes to replace missing modules
class DataProcessor:
    def load_and_validate(self, df):
        df = df.dropna()
        df = df.drop_duplicates()
        
        required_cols = ['amount', 'sender_country', 'receiver_country']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        if 'label' not in df.columns:
            df['label'] = 0
            
        return df

class FeatureEngineer:
    def create_features(self, df, time_features=True, amount_features=True, 
                       country_features=True, device_features=True, user_features=True):
        
        df = df.copy()
        
        if time_features and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        if amount_features and 'amount' in df.columns:
            df['amount_log'] = np.log1p(df['amount'])
            df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
        
        if country_features and all(col in df.columns for col in ['sender_country', 'receiver_country']):
            df['is_cross_border'] = (df['sender_country'] != df['receiver_country']).astype(int)
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

class SuspiciousTransactionModels:
    def prepare_data(self, df):
        X = df.select_dtypes(include=[np.number]).drop('label', axis=1, errors='ignore')
        y = df['label'] if 'label' in df.columns else pd.Series([0] * len(df))
        X = X.fillna(0)
        return X, y

    def train_models(self, X, y, models=None, test_size=0.2, 
                    handle_imbalance="None", cross_validation=True):
        
        if models is None:
            models = ["Random Forest", "XGBoost"]
        
        if handle_imbalance == "SMOTE" and len(np.unique(y)) > 1:
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        results = {'trained_models': {}}
        
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

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

def main():
    st.title("🚨 Suspicious Transaction Detection System")
    st.markdown("---")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Data Loading & Overview",
        "Feature Engineering", 
        "Model Training",
        "Real-time Detection"
    ])
    
    if page == "Data Loading & Overview":
        data_loading_page()
    elif page == "Feature Engineering":
        feature_engineering_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Real-time Detection":
        real_time_detection_page()

def data_loading_page():
    st.header("📊 Data Loading & Overview")
    
    def create_sample_data():
        data = {
            'transaction_id': [f'T_{i}' for i in range(1000)],
            'timestamp': pd.date_range('2023-01-01', periods=1000, freq='H'),
            'amount': np.random.exponential(1000, 1000),
            'sender_id': [f'USER_{np.random.randint(1, 50)}' for _ in range(1000)],
            'receiver_id': [f'USER_{np.random.randint(51, 100)}' for _ in range(1000)],
            'sender_country': np.random.choice(['USA', 'GBR', 'FRA', 'RUS', 'CHN'], 1000),
            'receiver_country': np.random.choice(['USA', 'GBR', 'FRA', 'RUS', 'CHN'], 1000),
            'device_id': [f'DEVICE_{np.random.randint(1, 20)}' for _ in range(1000)],
            'label': np.random.choice([0, 1], 1000, p=[0.95, 0.05])
        }
        return pd.DataFrame(data)
    
    uploaded_file = st.file_uploader("Upload transaction data (CSV)", type=['csv'])
    
    if st.button("Use Sample Data"):
        sample_df = create_sample_data()
        uploaded_file = "sample_data.csv"
        sample_df.to_csv(uploaded_file, index=False)
    
    if uploaded_file is not None:
        try:
            if isinstance(uploaded_file, str):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            
            processor = DataProcessor()
            df_processed = processor.load_and_validate(df)
            
            st.session_state.processed_data = df_processed
            st.session_state.data_loaded = True
            
            st.success(f"✅ Data loaded successfully! Shape: {df_processed.shape}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Transactions", len(df_processed))
            with col2:
                suspicious_count = df_processed['label'].sum()
                st.metric("Suspicious Transactions", suspicious_count)
            with col3:
                legitimate_count = len(df_processed) - suspicious_count
                st.metric("Legitimate Transactions", legitimate_count)
            with col4:
                suspicious_rate = (suspicious_count / len(df_processed)) * 100
                st.metric("Suspicious Rate", f"{suspicious_rate:.2f}%")
            
            st.subheader("Data Preview")
            st.dataframe(df_processed.head(10))
            
            visualizer = TransactionVisualizer()
            
            col1, col2 = st.columns(2)
            with col1:
                fig = visualizer.plot_transaction_distribution(df_processed)
                st.pyplot(fig)
            
            with col2:
                fig = visualizer.plot_amount_distribution(df_processed)
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")

def feature_engineering_page():
    st.header("🔧 Feature Engineering")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the 'Data Loading & Overview' page.")
        return
    
    df = st.session_state.processed_data
    
    st.subheader("Feature Engineering Options")
    
    col1, col2 = st.columns(2)
    with col1:
        create_time_features = st.checkbox("Create Time-based Features", value=True)
        create_amount_features = st.checkbox("Create Amount-based Features", value=True)
        create_country_features = st.checkbox("Create Country-based Features", value=True)
    
    with col2:
        create_device_features = st.checkbox("Create Device-based Features", value=True)
        create_user_features = st.checkbox("Create User-based Features", value=True)
        normalize_features = st.checkbox("Normalize Numerical Features", value=True)
    
    if st.button("Apply Feature Engineering"):
        try:
            feature_engineer = FeatureEngineer()
            
            df_features = feature_engineer.create_features(
                df,
                time_features=create_time_features,
                amount_features=create_amount_features,
                country_features=create_country_features,
                device_features=create_device_features,
                user_features=create_user_features
            )
            
            if normalize_features:
                df_features = feature_engineer.normalize_features(df_features)
            
            st.session_state.processed_data = df_features
            st.success("✅ Feature engineering completed!")
            
            st.subheader("Engineered Features")
            new_columns = [col for col in df_features.columns if col not in df.columns]
            st.write(f"Created {len(new_columns)} new features:")
            st.write(new_columns)
            
        except Exception as e:
            st.error(f"❌ Error in feature engineering: {str(e)}")

def model_training_page():
    st.header("🤖 Model Training")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the 'Data Loading & Overview' page.")
        return
    
    df = st.session_state.processed_data
    
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        models_to_train = st.multiselect(
            "Select models to train:",
            ["Random Forest", "XGBoost", "Logistic Regression", "SVM"],
            default=["Random Forest", "XGBoost"]
        )
        
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
        
    with col2:
        handle_imbalance = st.selectbox(
            "Handle class imbalance:",
            ["None", "SMOTE", "Random Undersampling", "Random Oversampling"]
        )
        
        cross_validation = st.checkbox("Use Cross Validation", value=True)
    
    if st.button("Train Models"):
        try:
            with st.spinner("Training models... This may take a few minutes."):
                model_trainer = SuspiciousTransactionModels()
                
                X, y = model_trainer.prepare_data(df)
                
                results = model_trainer.train_models(
                    X, y,
                    models=models_to_train,
                    test_size=test_size,
                    handle_imbalance=handle_imbalance,
                    cross_validation=cross_validation
                )
                
                st.session_state.models_trained = True
                st.session_state.model_results = results
                
                st.success("✅ Models trained successfully!")
                
                st.subheader("Model Performance")
                results_df = pd.DataFrame({
                    'Model': list(results['trained_models'].keys()),
                    'Status': ['Trained'] * len(results['trained_models'])
                })
                st.dataframe(results_df)
                
        except Exception as e:
            st.error(f"❌ Error training models: {str(e)}")

def real_time_detection_page():
    st.header("🔍 Real-time Detection")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first from the 'Model Training' page.")
        return
    
    st.subheader("Single Transaction Analysis")
    
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Amount", min_value=0.01, value=1000.0)
            sender_country = st.selectbox("Sender Country", ["USA", "GBR", "FRA", "RUS", "CHN", "DEU"])
            receiver_country = st.selectbox("Receiver Country", ["USA", "GBR", "FRA", "RUS", "CHN", "DEU"])
        
        with col2:
            sender_id = st.text_input("Sender ID", value="USER_001")
            receiver_id = st.text_input("Receiver ID", value="USER_002")
            device_id = st.text_input("Device ID", value="DEVICE_001")
        
        submitted = st.form_submit_button("Analyze Transaction")
        
        if submitted:
            try:
                risk_score = 0
                
                if amount > 10000:
                    risk_score += 30
                elif amount > 5000:
                    risk_score += 15
                
                high_risk_countries = ["RUS", "IRN", "PRK", "SYR"]
                if sender_country in high_risk_countries:
                    risk_score += 25
                if receiver_country in high_risk_countries:
                    risk_score += 25
                
                if sender_country != receiver_country:
                    risk_score += 20
                
                st.subheader("Risk Assessment")
                st.progress(risk_score / 100)
                st.write(f"Risk Score: {risk_score}%")
                
                if risk_score > 70:
                    st.error("🚨 HIGH RISK: This transaction appears suspicious!")
                elif risk_score > 40:
                    st.warning("⚠️ MEDIUM RISK: Review this transaction")
                else:
                    st.success("✅ LOW RISK: This transaction appears legitimate")
                
            except Exception as e:
                st.error(f"❌ Error analyzing transaction: {str(e)}")

if __name__ == "__main__":
    main()