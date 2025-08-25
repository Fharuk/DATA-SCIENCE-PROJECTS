# File: creditcard_fraud_streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ----------------------------
# Load model and supporting files
# ----------------------------
model = joblib.load("creditcard_fraud_lgbm_final.pkl")
explainer = joblib.load("shap_explainer_final.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ----------------------------
# Load new transactions
# ----------------------------
st.title("Credit Card Fraud Detection")
uploaded_file = st.file_uploader("Upload new credit card transactions CSV", type="csv")
if uploaded_file:
    new_transactions = pd.read_csv(uploaded_file)

    # Feature engineering
    new_transactions['Amount_log'] = np.log1p(new_transactions['Amount'])
    new_transactions['hour_of_day'] = (new_transactions['Time'] // 3600) % 24

    # ----------------------------
    # Threshold selection
    # ----------------------------
    threshold = st.slider("Select probability threshold for fraud detection", 0.1, 1.0, 0.95, 0.01)

    # Prediction function
    def predict_fraud(new_data: pd.DataFrame, threshold: float):
        X_new = new_data[feature_columns].copy()
        probs = model.predict_proba(X_new)[:, 1]
        preds = (probs >= threshold).astype(int)
        shap_values = explainer.shap_values(X_new)

        # Only keep SHAP values for class 1 if multi-output
        if isinstance(shap_values, list):
            shap_values_class1 = shap_values[1]
        else:
            shap_values_class1 = shap_values

        results = pd.DataFrame({
            "probability_fraud": probs,
            "predicted_class": preds
        }, index=new_data.index)

        return results, shap_values_class1

    results, shap_values_class1 = predict_fraud(new_transactions, threshold)
    new_transactions['predicted_class'] = results['predicted_class']
    new_transactions['probability_fraud'] = results['probability_fraud']

    predicted_frauds = new_transactions[new_transactions['predicted_class'] == 1]
    st.write(f"Predicted Frauds: {predicted_frauds.shape[0]}")

    if predicted_frauds.empty:
        st.warning("No frauds predicted at this threshold. Consider lowering the threshold.")
    else:
        fraud_indices = predicted_frauds.index.tolist()
        shap_values_frauds = shap_values_class1[fraud_indices]
        X_frauds = predicted_frauds[feature_columns]

        # ----------------------------
        # SHAP Summary Plot
        # ----------------------------
        st.subheader("SHAP Summary Plot for Predicted Frauds")
        fig_summary, ax_summary = plt.subplots(figsize=(10,6))
        shap.summary_plot(shap_values_frauds, X_frauds, plot_type="bar", show=False)
        st.pyplot(fig_summary)

        # ----------------------------
        # Individual Fraud SHAP Explanations
        # ----------------------------
        st.subheader("Individual Fraud SHAP Explanations")
        for i, idx in enumerate(fraud_indices, 1):
            st.write(f"Fraud #{i} - Probability: {predicted_frauds.loc[idx, 'probability_fraud']:.4f}")

            # Compute top 3 features
            top_3_idx = np.argsort(np.abs(shap_values_class1[idx]))[::-1][:3]
            top_3_features = [feature_columns[j] for j in top_3_idx]
            st.write(f"Top 3 contributing features: {top_3_features}")

            # Force plot
            fig_force, ax_force = plt.subplots(figsize=(12,3))
            shap.force_plot(
                explainer.expected_value,  # scalar expected_value
                shap_values_class1[idx],
                new_transactions.loc[idx, feature_columns],
                matplotlib=True,
                show=False
            )
            st.pyplot(fig_force)

        # ----------------------------
        # Table of all predicted frauds
        # ----------------------------
        st.subheader("All Predicted Frauds")
        predicted_frauds_display = predicted_frauds.copy()
        predicted_frauds_display['top_3_features'] = [
            [feature_columns[j] for j in np.argsort(np.abs(shap_values_class1[idx]))[::-1][:3]]
            for idx in fraud_indices
        ]
        st.dataframe(predicted_frauds_display[["Time", "Amount", "probability_fraud", "top_3_features"]])
