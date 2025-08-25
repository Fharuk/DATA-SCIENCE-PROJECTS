# wine_quality_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved models and scaler
rf_model = joblib.load('random_forest_wine_model.pkl')
xgb_model = joblib.load('xgboost_wine_model.pkl')
scaler = joblib.load('wine_scaler.pkl')

st.title("Vinho Verde Wine Quality Predictor 🍷")
st.write("""
This app predicts the quality of Portuguese 'Vinho Verde' wine based on physicochemical properties.
""")

# Sidebar for user input
st.sidebar.header("Input Wine Parameters")

def user_input_features():
    fixed_acidity = st.sidebar.slider('Fixed Acidity', 3.0, 16.0, 7.0)
    volatile_acidity = st.sidebar.slider('Volatile Acidity', 0.08, 1.6, 0.34)
    citric_acid = st.sidebar.slider('Citric Acid', 0.0, 1.7, 0.32)
    residual_sugar = st.sidebar.slider('Residual Sugar', 0.6, 66.0, 5.4)
    chlorides = st.sidebar.slider('Chlorides', 0.009, 0.61, 0.056)
    free_sulfur_dioxide = st.sidebar.slider('Free Sulfur Dioxide', 1, 290, 30)
    total_sulfur_dioxide = st.sidebar.slider('Total Sulfur Dioxide', 6, 440, 116)
    density = st.sidebar.slider('Density', 0.987, 1.039, 0.995)
    pH = st.sidebar.slider('pH', 2.72, 4.01, 3.22)
    sulphates = st.sidebar.slider('Sulphates', 0.22, 2.0, 0.53)
    alcohol = st.sidebar.slider('Alcohol', 8.0, 14.9, 10.5)
    type_white = st.sidebar.selectbox('Wine Type', ['Red', 'White'])
    
    # Encode type
    type_white_encoded = 1 if type_white == 'White' else 0
    
    data = {
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol,
        'type_white': type_white_encoded
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Scale input features
input_scaled = scaler.transform(input_df)

# Prediction buttons
st.subheader("Predicted Wine Quality")
if st.button("Predict with Random Forest"):
    pred_rf = rf_model.predict(input_scaled)
    st.write(f"Predicted Quality Score (Random Forest): {pred_rf[0]:.2f}")

if st.button("Predict with XGBoost"):
    pred_xgb = xgb_model.predict(input_scaled)
    st.write(f"Predicted Quality Score (XGBoost): {pred_xgb[0]:.2f}")
