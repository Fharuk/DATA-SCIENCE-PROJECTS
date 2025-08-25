# app_enhanced.py
import streamlit as st
import pandas as pd
import joblib
import json
import shap
import matplotlib.pyplot as plt

# --- Load Model and Features ---
model = joblib.load("catboost_smartphone_price_model.pkl")
with open("feature_list.json", "r") as f:
    feature_list = json.load(f)

# --- App Title ---
st.title("Smartphone Price Prediction 🚀")
st.write("Predict smartphone prices based on specifications and see feature importance!")

# --- Sidebar Inputs ---
st.sidebar.header("Enter Smartphone Specifications")

def user_input_features():
    data = {}
    data['brand_name_oneplus'] = st.sidebar.checkbox("Brand: OnePlus")
    data['brand_name_samsung'] = st.sidebar.checkbox("Brand: Samsung")
    data['brand_name_motorola'] = st.sidebar.checkbox("Brand: Motorola")
    data['brand_name_realme'] = st.sidebar.checkbox("Brand: Realme")
    
    data['processor_brand_snapdragon'] = st.sidebar.checkbox("Processor: Snapdragon")
    data['processor_brand_dimensity'] = st.sidebar.checkbox("Processor: Dimensity")
    data['processor_brand_exynos'] = st.sidebar.checkbox("Processor: Exynos")
    
    data['os_android'] = st.sidebar.checkbox("OS: Android", True)
    data['os_ios'] = st.sidebar.checkbox("OS: iOS")
    
    # Numerical Features
    data['rating'] = st.sidebar.slider("Rating (0-100)", 60, 90, 80)
    data['has_5g'] = st.sidebar.checkbox("5G Support", True)
    data['has_nfc'] = st.sidebar.checkbox("NFC Support", False)
    data['has_ir_blaster'] = st.sidebar.checkbox("IR Blaster", False)
    data['num_cores'] = st.sidebar.slider("Number of CPU Cores", 4, 8, 8)
    data['processor_speed'] = st.sidebar.slider("Processor Speed (GHz)", 1.2, 3.2, 2.5)
    data['battery_capacity'] = st.sidebar.slider("Battery Capacity (mAh)", 1800, 22000, 5000)
    data['fast_charging_available'] = st.sidebar.checkbox("Fast Charging Available", True)
    data['fast_charging'] = st.sidebar.slider("Fast Charging (W)", 0, 120, 33)
    data['ram_capacity'] = st.sidebar.slider("RAM (GB)", 1, 18, 6)
    data['internal_memory'] = st.sidebar.slider("Internal Memory (GB)", 8, 1024, 128)
    data['screen_size'] = st.sidebar.slider("Screen Size (inches)", 3.5, 8.0, 6.5)
    data['refresh_rate'] = st.sidebar.slider("Refresh Rate (Hz)", 60, 240, 120)
    data['num_rear_cameras'] = st.sidebar.slider("Rear Cameras", 1, 4, 3)
    data['num_front_cameras'] = st.sidebar.slider("Front Cameras", 1, 2, 1)
    data['extended_memory_available'] = st.sidebar.checkbox("Extended Memory Available", True)
    data['extended_upto'] = st.sidebar.slider("Extended Memory Upto (GB)", 0, 2048, 512)
    data['ppi'] = st.sidebar.slider("Pixel Density (PPI)", 480, 3840, 1080)
    data['performance_score'] = st.sidebar.slider("Performance Score", 1, 200, 50)
    data['camera_strength'] = st.sidebar.slider("Camera Strength", 1, 200, 50)
    
    # Fill missing features with 0
    for feat in feature_list:
        if feat not in data:
            data[feat] = 0
    return pd.DataFrame([data])

# --- Single Prediction ---
st.subheader("Single Smartphone Prediction")
input_df = user_input_features()
predicted_price = model.predict(input_df)[0]
st.write(f"💰 Predicted Price: ₹ {predicted_price:,.0f}")

# SHAP explanation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_df)
st.subheader("Top Feature Contributions for This Prediction")
fig, ax = plt.subplots()
shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], input_df.iloc[0], show=False)
st.pyplot(fig)

# --- Batch Predictions ---
st.subheader("Batch Predictions from CSV")
uploaded_file = st.file_uploader("Upload CSV file with smartphone specs", type=["csv"])
if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    # Ensure all features exist
    for feat in feature_list:
        if feat not in batch_df.columns:
            batch_df[feat] = 0
    batch_df = batch_df[feature_list]
    batch_predictions = model.predict(batch_df)
    batch_df['Predicted_Price'] = batch_predictions
    st.write(batch_df)
    # Download results
    csv = batch_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")

# --- Global SHAP Feature Importance ---
st.subheader("Global Feature Importance")
shap_values_all = explainer.shap_values(input_df)  # Can use entire training/test set if desired
fig2, ax2 = plt.subplots()
shap.summary_plot(shap_values_all, input_df, plot_type="bar", show=False)
st.pyplot(fig2)
