import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="❤️ Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.markdown("This app predicts the likelihood of heart disease using two ML models: **Random Forest (Subset A)** and **XGBoost (Subset B)**.")

# =============================
# Load Models & Scalers Safely
# =============================
def load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"⚠️ Failed to load {path}: {e}")
        return None

rf_model = load_pickle("rf_model_subset_A.pkl")
scaler_A = load_pickle("scaler_subset_A.pkl")

xgb_model = load_pickle("xgb_model_subset_B.pkl")
scaler_B = load_pickle("scaler_subset_B.pkl")

# =============================
# Define Features (based on training)
# =============================
features_A = ["age", "sex", "cp", "trestbps", "chol", "fbs", 
              "restecg", "thalach", "exang", "oldpeak", "slope", 
              "ca", "thal", "smoking", "diabetes", "bmi"]

features_B = ["age", "sex", "cp", "trestbps", "chol", 
              "thalach", "exang", "oldpeak", "slope", 
              "ca", "thal", "smoking", "diabetes", "bmi"]

# =============================
# Sidebar Inputs
# =============================
st.sidebar.header("📝 Patient Information")

age = st.sidebar.number_input("Age", 18, 100, 45)
sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.sidebar.slider("Chest Pain Type (cp)", 0, 3, 1)
trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.number_input("Cholesterol", 100, 400, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.sidebar.slider("Resting ECG (restecg)", 0, 2, 1)
thalach = st.sidebar.number_input("Max Heart Rate (thalach)", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.sidebar.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
slope = st.sidebar.slider("Slope", 0, 2, 1)
ca = st.sidebar.slider("Number of Major Vessels (ca)", 0, 4, 0)
thal = st.sidebar.slider("Thalassemia (thal)", 0, 3, 2)

# ✅ New fields added
smoking = st.sidebar.selectbox("Smoking", [0, 1])
diabetes = st.sidebar.selectbox("Diabetes", [0, 1])
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)

# =============================
# Collect Inputs into Dict
# =============================
input_dict = {
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
    "chol": chol, "fbs": fbs, "restecg": restecg,
    "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
    "slope": slope, "ca": ca, "thal": thal,
    "smoking": smoking, "diabetes": diabetes, "bmi": bmi
}

input_df = pd.DataFrame([input_dict])

# =============================
# Ensure Features Match Models
# =============================
for feat in features_A:
    if feat not in input_df.columns:
        input_df[feat] = 0
for feat in features_B:
    if feat not in input_df.columns:
        input_df[feat] = 0

input_A = input_df[features_A]
input_B = input_df[features_B]

# =============================
# Predict Button
# =============================
if st.button("🔍 Predict Heart Disease"):
    st.subheader("🔎 Prediction Results")

    # --- Subset A ---
    if rf_model and scaler_A:
        try:
            input_A_scaled = scaler_A.transform(input_A.values)
            pred_A = rf_model.predict(input_A_scaled)[0]
            prob_A = rf_model.predict_proba(input_A_scaled)[0][1]
            st.success(f"✅ Random Forest Prediction: {'Heart Disease' if pred_A==1 else 'No Heart Disease'} (Risk: {prob_A:.2f})")
        except Exception as e:
            st.error(f"⚠️ Random Forest prediction failed: {e}")
    else:
        st.warning("⚠️ Random Forest model not loaded properly.")

    # --- Subset B ---
    if xgb_model and scaler_B:
        try:
            input_B_scaled = scaler_B.transform(input_B.values)
            pred_B = xgb_model.predict(input_B_scaled)[0]
            prob_B = xgb_model.predict_proba(input_B_scaled)[0][1]
            st.success(f"✅ XGBoost Prediction: {'Heart Disease' if pred_B==1 else 'No Heart Disease'} (Risk: {prob_B:.2f})")
        except Exception as e:
            st.error(f"⚠️ XGBoost prediction failed: {e}")
    else:
        st.warning("⚠️ XGBoost model not loaded properly.")
