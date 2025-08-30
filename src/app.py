import streamlit as st
import pandas as pd
import numpy as np
import os
from src.data_processor import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.models import SuspiciousTransactionModels
from src.visualizations import TransactionVisualizer
import joblib
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Suspicious Transaction Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Data Loading & Overview",
        "Feature Engineering",
        "Model Training",
        "Transaction Analysis",
        "Real-time Detection"
    ])
    
    if page == "Data Loading & Overview":
        data_loading_page()
    elif page == "Feature Engineering":
        feature_engineering_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Transaction Analysis":
        transaction_analysis_page()
    elif page == "Real-time Detection":
        real_time_detection_page()

def data_loading_page():
    st.header("📊 Data Loading & Overview")
    
    # File upload
    uploaded_file = st.file_uploader("Upload transaction data (CSV)", type=['csv'])
    
    # Option to use sample data
    if st.button("Use Sample Data"):
        uploaded_file = "data/sample_input.csv"
    
    if uploaded_file is not None:
        try:
            # Load data
            if isinstance(uploaded_file, str):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            
            # Process data
            processor = DataProcessor()
            df_processed = processor.load_and_validate(df)
            
            st.session_state.processed_data = df_processed
            st.session_state.data_loaded = True
            
            st.success(f"✅ Data loaded successfully! Shape: {df_processed.shape}")
            
            # Display basic statistics
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
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df_processed.head(10))
            
            # Basic visualizations
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
    
    # Feature engineering options
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
            
            # Apply selected feature engineering
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
            
            # Display new features
            st.subheader("Engineered Features")
            new_columns = [col for col in df_features.columns if col not in df.columns]
            st.write(f"Created {len(new_columns)} new features:")
            st.write(new_columns)
            
            # Feature correlation heatmap
            visualizer = TransactionVisualizer()
            fig = visualizer.plot_correlation_heatmap(df_features)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"❌ Error in feature engineering: {str(e)}")

def model_training_page():
    st.header("🤖 Model Training")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the 'Data Loading & Overview' page.")
        return
    
    df = st.session_state.processed_data
    
    # Model selection
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
                
                # Prepare data
                X, y = model_trainer.prepare_data(df)
                
                # Train models
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
                
                # Display results
                st.subheader("Model Performance")
                
                # Create results dataframe
                results_df = pd.DataFrame(results['performance_metrics'])
                st.dataframe(results_df)
                
                # Plot model comparison
                visualizer = TransactionVisualizer()
                fig = visualizer.plot_model_comparison(results['performance_metrics'])
                st.pyplot(fig)
                
                # Feature importance
                if 'feature_importance' in results:
                    st.subheader("Feature Importance")
                    for model_name, importance in results['feature_importance'].items():
                        st.write(f"**{model_name}**")
                        fig = visualizer.plot_feature_importance(importance, model_name)
                        st.pyplot(fig)
                
        except Exception as e:
            st.error(f"❌ Error training models: {str(e)}")

def transaction_analysis_page():
    st.header("📈 Transaction Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the 'Data Loading & Overview' page.")
        return
    
    df = st.session_state.processed_data
    visualizer = TransactionVisualizer()
    
    # Analysis options
    st.subheader("Analysis Options")
    
    analysis_type = st.selectbox(
        "Select analysis type:",
        ["Transaction Patterns", "Country Analysis", "Amount Analysis", "Time Analysis", "Device Analysis"]
    )
    
    if analysis_type == "Transaction Patterns":
        col1, col2 = st.columns(2)
        with col1:
            fig = visualizer.plot_transaction_by_country(df)
            st.pyplot(fig)
        with col2:
            fig = visualizer.plot_cross_border_analysis(df)
            st.pyplot(fig)
    
    elif analysis_type == "Country Analysis":
        # Country-based analysis
        country_stats = df.groupby('sender_country').agg({
            'amount': ['mean', 'median', 'sum'],
            'label': ['count', 'sum']
        }).round(2)
        
        st.subheader("Country Statistics")
        st.dataframe(country_stats)
        
        fig = visualizer.plot_suspicious_by_country(df)
        st.pyplot(fig)
    
    elif analysis_type == "Amount Analysis":
        col1, col2 = st.columns(2)
        with col1:
            fig = visualizer.plot_amount_distribution(df)
            st.pyplot(fig)
        with col2:
            fig = visualizer.plot_amount_vs_suspicious(df)
            st.pyplot(fig)
    
    elif analysis_type == "Time Analysis":
        if 'hour' in df.columns:
            fig = visualizer.plot_hourly_patterns(df)
            st.pyplot(fig)
        else:
            st.info("Time features not available. Please run feature engineering first.")
    
    elif analysis_type == "Device Analysis":
        device_stats = df.groupby('device_id').agg({
            'transaction_id': 'count',
            'amount': 'sum',
            'label': 'sum'
        }).sort_values('transaction_id', ascending=False).head(10)
        
        st.subheader("Top 10 Devices by Transaction Count")
        st.dataframe(device_stats)

def real_time_detection_page():
    st.header("🔍 Real-time Detection")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first from the 'Model Training' page.")
        return
    
    st.subheader("Single Transaction Analysis")
    
    # Input form for transaction details
    with st.form("transaction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            amount = st.number_input("Amount", min_value=0.01, value=1000.0)
            sender_country = st.selectbox("Sender Country", ["USA", "GBR", "FRA", "RUS", "CHN", "DEU"])
            receiver_country = st.selectbox("Receiver Country", ["USA", "GBR", "FRA", "RUS", "CHN", "DEU"])
        
        with col2:
            sender_id = st.text_input("Sender ID", value="USER_001")
            receiver_id = st.text_input("Receiver ID", value="USER_002")
            device_id = st.text_input("Device ID", value="DEVICE_001")
        
        with col3:
            timestamp = st.datetime_input("Timestamp", value=datetime.now())
            transaction_id = st.text_input("Transaction ID", value="T_NEW_001")
        
        submitted = st.form_submit_button("Analyze Transaction")
        
        if submitted:
            # Create transaction dataframe
            transaction_data = {
                'transaction_id': [transaction_id],
                'timestamp': [timestamp.isoformat()],
                'amount': [amount],
                'sender_id': [sender_id],
                'receiver_id': [receiver_id],
                'sender_country': [sender_country],
                'receiver_country': [receiver_country],
                'device_id': [device_id],
                'label': [0]  # Placeholder
            }
            
            transaction_df = pd.DataFrame(transaction_data)
            
            try:
                # Process the transaction through the same pipeline
                feature_engineer = FeatureEngineer()
                processed_transaction = feature_engineer.create_features(transaction_df)
                
                # Load models and make predictions
                results = st.session_state.model_results
                predictions = {}
                
                for model_name, model_info in results['trained_models'].items():
                    model = model_info['model']
                    
                    # Prepare features (ensure same columns as training)
                    X_columns = model_info.get('feature_columns', [])
                    X_transaction = processed_transaction[X_columns].fillna(0)
                    
                    # Make prediction
                    prediction = model.predict(X_transaction)[0]
                    probability = model.predict_proba(X_transaction)[0]
                    
                    predictions[model_name] = {
                        'prediction': prediction,
                        'probability': probability[1]  # Probability of being suspicious
                    }
                
                # Display results
                st.subheader("Detection Results")
                
                for model_name, pred_info in predictions.items():
                    is_suspicious = pred_info['prediction'] == 1
                    confidence = pred_info['probability'] * 100
                    
                    status = "🚨 SUSPICIOUS" if is_suspicious else "✅ LEGITIMATE"
                    color = "red" if is_suspicious else "green"
                    
                    st.markdown(f"**{model_name}**: {status}")
                    st.progress(confidence / 100)
                    st.write(f"Confidence: {confidence:.2f}%")
                    st.markdown("---")
                
                # Average prediction
                avg_probability = np.mean([pred['probability'] for pred in predictions.values()])
                avg_prediction = avg_probability > 0.5
                
                st.subheader("Ensemble Prediction")
                status = "🚨 SUSPICIOUS" if avg_prediction else "✅ LEGITIMATE"
                st.markdown(f"**Overall Assessment**: {status}")
                st.markdown(f"**Average Confidence**: {avg_probability * 100:.2f}%")
                
            except Exception as e:
                st.error(f"❌ Error analyzing transaction: {str(e)}")

if __name__ == "__main__":
    main()
