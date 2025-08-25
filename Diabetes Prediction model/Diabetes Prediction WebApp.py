# ===========================================================
#  Diabetes Prediction Web App (Interactive Dashboard)
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load trained pipeline
pipeline = joblib.load("diabetes_pipeline.pkl")

# Load feature names and importances from trained model
xgb_model = pipeline.named_steps["classifier"]
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = xgb_model.feature_importances_

# Feature importance dataframe
fi_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# ===========================
# App Title
# ===========================
st.title("🩺 Diabetes Prediction Dashboard")
st.write("Predict diabetes risk, adjust thresholds, and explore patient risk visually.")

# ===========================
# Adjustable threshold
# ===========================
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider(
    "Classification Probability Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)
st.sidebar.write(f"Patients with predicted probability ≥ {threshold:.2f} will be flagged as High Risk.")

# ===========================
# Single Patient Prediction
# ===========================
st.header("🔹 Single Patient Prediction")
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.slider("Age", 1, 120, 30)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
smoking_history = st.selectbox("Smoking History", ["never", "former", "current", "ever", "not current", "No Info"])
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
HbA1c_level = st.slider("HbA1c Level", 3.0, 15.0, 5.5)
blood_glucose_level = st.slider("Blood Glucose Level", 50, 300, 100)

if st.button("🔍 Predict Single Patient"):
    patient_data = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": HbA1c_level,
        "blood_glucose_level": blood_glucose_level
    }])
    
    probability = pipeline.predict_proba(patient_data)[0][1]
    prediction = int(probability >= threshold)

    if prediction == 1:
        st.error(f" High Risk of Diabetes (Probability: {probability:.2f})")
    else:
        st.success(f" No Diabetes Risk Detected (Probability: {probability:.2f})")

# ===========================
# Batch Prediction via CSV
# ===========================
st.header("🔹 Batch Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload CSV file with patient data", type=["csv"])

if uploaded_file is not None:
    df_batch = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.dataframe(df_batch.head())

    if st.button("🔍 Predict Batch"):
        probabilities = pipeline.predict_proba(df_batch)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        
        df_batch["Predicted_Probability"] = probabilities
        df_batch["Predicted_Label"] = predictions

        # Summary statistics
        total_patients = len(df_batch)
        high_risk_count = df_batch["Predicted_Label"].sum()
        mean_probability = df_batch["Predicted_Probability"].mean()

        st.subheader("📊 Summary")
        st.write(f"Total Patients: {total_patients}")
        st.write(f"Patients at High Risk: {high_risk_count}")
        st.write(f"Average Predicted Probability: {mean_probability:.2f}")

        # Display predictions
        st.subheader("✅ Patient Predictions")
        st.dataframe(df_batch)

        # Probability distribution colored by risk
        st.subheader("📈 Diabetes Risk Probability Distribution")
        plt.figure(figsize=(8,4))
        sns.histplot(df_batch, x="Predicted_Probability", hue="Predicted_Label",
                     palette={0: "green", 1: "red"}, bins=20, kde=True)
        plt.xlabel("Predicted Probability")
        plt.ylabel("Number of Patients")
        plt.legend(title="Risk", labels=["Low Risk", "High Risk"])
        st.pyplot(plt)

        # Feature Importance Plot
        st.subheader(" Feature Importance (XGBoost)")
        plt.figure(figsize=(8,6))
        sns.barplot(x="Importance", y="Feature", data=fi_df.head(10), palette="viridis")
        plt.title("Top 10 Features Contributing to Diabetes Prediction")
        st.pyplot(plt)

        # Download CSV
        csv = df_batch.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv,
            file_name="diabetes_predictions.csv",
            mime="text/csv"
        )
