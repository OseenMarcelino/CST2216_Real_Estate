"""
Real Estate Price Prediction Web Application
This application uses machine learning models to predict real estate prices.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.data_loader import load_data, prepare_features, get_data_info
from src.model import split_data, train_linear_regression, train_random_forest, make_prediction
from src.evaluation import evaluate_model, compare_models
from src.utils import logger


# Page configuration
st.set_page_config(
    page_title="Real Estate Price Prediction",
    page_icon="house",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Real Estate Price Prediction")
st.markdown("Use machine learning to predict real estate prices based on property features.")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select a page:", 
    ["Dashboard", "Model Training", "Price Predictor"])

# Load and prepare data
@st.cache_resource
def load_and_prepare_data():
    """Load and prepare data with caching"""
    try:
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'final.csv')
        df = load_data(data_path)
        X, y = prepare_features(df)
        logger.info("Data loaded and prepared successfully")
        return df, X, y
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Data loading failed: {str(e)}")
        return None, None, None


# Load data
df, X, y = load_and_prepare_data()

if df is None:
    st.error("Failed to load data. Please check the data file.")
    st.stop()


# PAGE 1: Dashboard
if page == "Dashboard":
    st.header("Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Properties", len(df))
    
    with col2:
        st.metric("Average Price", f"${df['price'].mean():,.0f}")
    
    with col3:
        st.metric("Lowest Price", f"${df['price'].min():,.0f}")
    
    with col4:
        st.metric("Highest Price", f"${df['price'].max():,.0f}")
    
    st.divider()
    
    # Display data sample
    st.subheader("Data Sample")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.divider()
    
    # Price distribution
    st.subheader("Price Distribution")
    st.bar_chart(df['price'].value_counts().sort_index(), height=400)
    
    # Feature statistics
    st.subheader("Feature Statistics")
    stats_df = df.describe().T
    st.dataframe(stats_df, use_container_width=True)


# PAGE 2: Model Training
elif page == "Model Training":
    st.header("Model Training & Evaluation")
    
    st.info("This page trains two machine learning models and compares their performance.")
    
    # Model training section
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
    
    with col2:
        train_button = st.button("Train Models", type="primary")
    
    if train_button:
        with st.spinner("Training models... Please wait."):
            try:
                # Split data
                X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
                
                # Train models
                lr_model = train_linear_regression(X_train, y_train)
                rf_model = train_random_forest(X_train, y_train)
                
                # Evaluate models
                lr_results = evaluate_model(lr_model, X_train, y_train, X_test, y_test, "Linear Regression")
                rf_results = evaluate_model(rf_model, X_train, y_train, X_test, y_test, "Random Forest")
                
                # Store in session state
                st.session_state.lr_model = lr_model
                st.session_state.rf_model = rf_model
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.lr_results = lr_results
                st.session_state.rf_results = rf_results
                
                st.success("Models trained successfully!")
                logger.info("Models trained in Streamlit app")
                
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
                logger.error(f"Training error: {str(e)}")
    
    # Display results if available
    if 'lr_results' in st.session_state and 'rf_results' in st.session_state:
        st.divider()
        st.subheader("Model Performance Comparison")
        
        # Create comparison table
        comparison_data = {
            'Model': ['Linear Regression', 'Random Forest'],
            'Train MAE': [
                f"${st.session_state.lr_results['train']['MAE']:,.2f}",
                f"${st.session_state.rf_results['train']['MAE']:,.2f}"
            ],
            'Test MAE': [
                f"${st.session_state.lr_results['test']['MAE']:,.2f}",
                f"${st.session_state.rf_results['test']['MAE']:,.2f}"
            ],
            'Train R²': [
                f"{st.session_state.lr_results['train']['R2']:.4f}",
                f"{st.session_state.rf_results['train']['R2']:.4f}"
            ],
            'Test R²': [
                f"{st.session_state.lr_results['test']['R2']:.4f}",
                f"{st.session_state.rf_results['test']['R2']:.4f}"
            ]
        }
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
        # Model recommendation
        lr_mae = st.session_state.lr_results['test']['MAE']
        rf_mae = st.session_state.rf_results['test']['MAE']
        
        if rf_mae < lr_mae:
            st.success(f"**Random Forest Model** performs better with Test MAE of ${rf_mae:,.2f}")
        else:
            st.success(f"**Linear Regression Model** performs better with Test MAE of ${lr_mae:,.2f}")


# PAGE 3: Price Predictor
elif page == "Price Predictor":
    st.header("Predict Real Estate Price")
    
    st.info("Select property features to predict its price using the trained model.")
    
    # Check if model is available
    if 'rf_model' not in st.session_state:
        st.warning("Please train the models first on the 'Model Training' page.")
    else:
        # Create input columns
        st.subheader("Property Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            year_sold = st.number_input("Year Sold", min_value=1950, max_value=2025, value=2013)
            property_tax = st.number_input("Property Tax ($)", min_value=0, value=234)
            insurance = st.number_input("Insurance ($)", min_value=0, value=81)
            beds = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
            basement = st.selectbox("Has Basement?", [0, 1])
            recession = st.selectbox("During Recession?", [0, 1])
        
        with col2:
            baths = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
            sqft = st.number_input("Square Feet", min_value=100, value=1500)
            year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2000)
            lot_size = st.number_input("Lot Size (sq ft)", min_value=0, value=5000)
            popular = st.selectbox("Popular Area?", [0, 1])
            property_type = st.selectbox("Property Type", [0, 1], format_func=lambda x: "House" if x == 0 else "Condo")
        
        # Calculate property age
        property_age = 2025 - year_built
        
        # Prediction button
        if st.button("Predict Price", type="primary"):
            try:
                # Create input dataframe
                input_data = pd.DataFrame({
                    'year_sold': [year_sold],
                    'property_tax': [property_tax],
                    'insurance': [insurance],
                    'beds': [beds],
                    'baths': [baths],
                    'sqft': [sqft],
                    'year_built': [year_built],
                    'lot_size': [lot_size],
                    'basement': [basement],
                    'popular': [popular],
                    'recession': [recession],
                    'property_age': [property_age],
                    'property_type_Condo': [property_type]
                })
                
                # Make predictions with both models
                lr_prediction = st.session_state.lr_model.predict(input_data)[0]
                rf_prediction = st.session_state.rf_model.predict(input_data)[0]
                
                logger.info(f"Price prediction made - LR: ${lr_prediction:,.2f}, RF: ${rf_prediction:,.2f}")
                
                # Display predictions
                st.divider()
                st.subheader("Predicted Prices")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Linear Regression", f"${lr_prediction:,.2f}")
                
                with col2:
                    st.metric("Random Forest", f"${rf_prediction:,.2f}")
                
                # Average prediction
                avg_prediction = (lr_prediction + rf_prediction) / 2
                st.markdown(f"### Average Prediction: **${avg_prediction:,.2f}**")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                logger.error(f"Prediction error: {str(e)}")


# Footer
st.divider()
st.markdown("""
---
**Real Estate Price Prediction Web App** | Built with Streamlit & Machine Learning
""")
