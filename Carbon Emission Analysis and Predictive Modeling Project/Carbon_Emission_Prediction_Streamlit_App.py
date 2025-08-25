import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# -------------------------------
# Load trained model and features
# -------------------------------
MODEL_PATH = r"C:\Users\AKINMADE FARUQ\Downloads\PROJECT MATERIALS\My Projects\GITHUB\Carbon emission\saved_models\xgboost_carbon_model.pkl"
FEATURES_PATH = r"C:\Users\AKINMADE FARUQ\Downloads\PROJECT MATERIALS\My Projects\GITHUB\Carbon emission\saved_models\numeric_features.pkl"


# Load model
with open(MODEL_PATH, "rb") as f:
    best_model = pickle.load(f)

# Load numeric features (from preprocessing step)
with open(FEATURES_PATH, "rb") as f:
    numeric_features = pickle.load(f)

# Extract full feature names that the model expects
model_feature_names = best_model.get_booster().feature_names

st.title("🌍 Carbon Emission Prediction App")
st.write("Enter input values to predict carbon emissions using the trained XGBoost model.")

# -------------------------------
# Define user input fields
# -------------------------------
user_inputs = {}

# Numeric input fields
for feature in numeric_features:
    user_inputs[feature] = st.number_input(f"{feature}", value=0.0)

# Example categorical inputs (you can expand as needed)
sector = st.selectbox("Industry Sector", ["Energy", "IT", "Manufacturing", "Transport"])
transport_mode = st.selectbox("Supply Chain Transport Mode", ["Rail", "Ship", "Truck"])
strategy = st.selectbox("Carbon Reduction Strategy", ["Efficiency Upgrade", "Process Reengineering", "Renewable Adoption"])
industry = st.selectbox("Industry Sub-sector", ["Cement Production", "Steel Manufacturing", "Logistics"])

# -------------------------------
# Convert input to dataframe
# -------------------------------
input_df = pd.DataFrame([user_inputs])

# Handle categorical encoding (manual one-hot encoding)
categorical_features = {
    "Sector": sector,
    "Supply_Chain_Transport_Mode": transport_mode,
    "Carbon_Reduction_Strategy": strategy,
    "Industry_Sectors": industry,
}

for cat, value in categorical_features.items():
    col_name = f"{cat}_{value}"
    input_df[col_name] = 1

# Ensure all model expected features exist
for col in model_feature_names:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder to match training features
input_df = input_df[model_feature_names]

st.write("### Processed Input for Prediction")
st.dataframe(input_df)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Carbon Emission"):
    prediction = best_model.predict(input_df.astype(np.float32))[0]
    st.success(f"🌱 Predicted Carbon Emission: {prediction:.2f}")
