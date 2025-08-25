import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Load pipeline
# -------------------------------
pipeline = joblib.load("pipeline_with_footfall_log.joblib")

# Features expected by the pipeline
pipeline_features = ['footfall', 'tempMode', 'AQ', 'USS', 'CS', 'VOC', 'RP', 'IP', 'Temperature']

# -------------------------------
# App Title
# -------------------------------
st.title("Predictive Failure Detection System")
st.write("Upload a CSV file or use the sliders to predict system failure.")

# -------------------------------
# CSV Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Keep only the raw features
    missing_cols = set(pipeline_features) - set(df.columns)
    if missing_cols:
        st.error(f"The following required columns are missing in the CSV: {missing_cols}")
    else:
        df = df[pipeline_features]
        predictions = pipeline.predict(df)
        df['fail_prediction'] = predictions
        st.write("Predictions:")
        st.dataframe(df)
        # Optionally allow download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")

# -------------------------------
# Single Input via Sliders
# -------------------------------
st.sidebar.header("Single Input Sliders")
single_input = {}
for feat in pipeline_features:
    if feat in ['tempMode', 'CS', 'IP']:  # categorical-like
        min_val = 0
        max_val = 10
        step = 1
    else:  # numeric features
        min_val = 0
        max_val = 1000
        step = 1
    single_input[feat] = st.sidebar.slider(feat, min_value=min_val, max_value=max_val, value=min_val, step=step)

single_df = pd.DataFrame([single_input])
# Keep only raw features
single_df = single_df[pipeline_features]

# Prediction
pred = pipeline.predict(single_df)[0]
st.sidebar.write(f"Predicted Failure: {pred}")
