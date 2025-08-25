# ===============================
# 🎓 Student Performance Prediction Dashboard
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

# -------------------------------
# 1️⃣ App Layout
# -------------------------------
st.title("🎓 Student Performance Prediction & Insights")

# -------------------------------
# 2️⃣ Upload CSV Dataset
# -------------------------------
uploaded_file = st.file_uploader("Upload Student Dataset CSV", type=["csv"])

if uploaded_file:
    st.success("Dataset loaded successfully!")
    df = pd.read_csv(uploaded_file)
    st.write("Sample data: ", df.head())
    
    # -------------------------------
    # 3️⃣ Preprocessing
    # -------------------------------
    st.write("🔹 Preprocessing Dataset...")
    
    # Example preprocessing (adjust based on your original preprocessing)
    # Encode categorical variables, fill missing values, etc.
    df_processed = df.copy()
    
    # Example one-hot encoding for categorical features
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    st.success(f"Preprocessing complete. Feature matrix shape: {df_processed.shape}")

    # -------------------------------
    # 4️⃣ Load Pre-trained XGBoost Model
    # -------------------------------
    st.write("🔹 Loading Pre-trained XGBoost Model...")
    try:
        xgb_model = joblib.load("xgb_student_model.pkl")
        st.success("XGBoost model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load XGBoost model: {e}")
        st.stop()

    # -------------------------------
    # 5️⃣ Align Features
    # -------------------------------
    # Feature names used during training
    feature_names_train = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion', 
                           'ParentAnsweringSurvey', 'ParentschoolSatisfaction', 'StudentAbsenceDays',
                           'NationalITy_Iran','NationalITy_Iraq','NationalITy_Jordan','NationalITy_KW',
                           'NationalITy_Lybia','NationalITy_Morocco','NationalITy_Palestine',
                           'NationalITy_SaudiArabia','NationalITy_Syria','NationalITy_Tunis','NationalITy_USA',
                           'NationalITy_lebanon','NationalITy_venzuela','PlaceofBirth_Iran','PlaceofBirth_Iraq',
                           'PlaceofBirth_Jordan','PlaceofBirth_KuwaIT','PlaceofBirth_Lybia','PlaceofBirth_Morocco',
                           'PlaceofBirth_Palestine','PlaceofBirth_SaudiArabia','PlaceofBirth_Syria','PlaceofBirth_Tunis',
                           'PlaceofBirth_USA','PlaceofBirth_lebanon','PlaceofBirth_venzuela','StageID_MiddleSchool',
                           'StageID_lowerlevel','GradeID_G-04','GradeID_G-05','GradeID_G-06','GradeID_G-07','GradeID_G-08',
                           'GradeID_G-09','GradeID_G-10','GradeID_G-11','GradeID_G-12','SectionID_B','SectionID_C',
                           'Topic_Biology','Topic_Chemistry','Topic_English','Topic_French','Topic_Geology',
                           'Topic_History','Topic_IT','Topic_Math','Topic_Quran','Topic_Science','Topic_Spanish',
                           'Semester_S','Relation_Mum','gender_M','Engagement_Score','Parental_Support_Score']

    # Add missing columns
    for col in feature_names_train:
        if col not in df_processed.columns:
            df_processed[col] = 0

    # Reorder columns
    X = df_processed[feature_names_train]

    # -------------------------------
    # 6️⃣ Predictions & Risk Segmentation
    # -------------------------------
    st.write("🔹 Predicting Student Performance & Segmenting Risk...")
    pred_numeric = xgb_model.predict(X)
    pred_labels = pd.Series(pred_numeric).map({0:'Low',1:'Medium',2:'High'})
    risk_mapping = {'Low':'At-Risk','Medium':'Moderate-Risk','High':'Safe'}
    risk_categories = pred_labels.map(risk_mapping)

    df_predictions = df.copy()
    df_predictions['Predicted_Class'] = pred_labels
    df_predictions['Risk_Category'] = risk_categories

    st.write("Sample predictions:")
    st.dataframe(df_predictions.head())

    # -------------------------------
    # 7️⃣ Model Explainability (SHAP)
    # -------------------------------
    st.write("🔹 Model Explainability (SHAP)")

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)

    # ---- 7a. SHAP Summary Plot ----
    st.write("**SHAP Summary Plot (All Students)**")
    fig_summary, ax_summary = plt.subplots(figsize=(10,6))
    shap.summary_plot(shap_values, X, class_names=['Low','Medium','High'], show=False)
    st.pyplot(fig_summary)

    # ---- 7b. SHAP Feature Importance (Bar Plot) ----
    st.write("**SHAP Feature Importance (Bar Plot)**")
    fig_bar, ax_bar = plt.subplots(figsize=(10,6))
    shap.summary_plot(shap_values, X, plot_type="bar", class_names=['Low','Medium','High'], show=False)
    st.pyplot(fig_bar)

    # -------------------------------
    # 8️⃣ Students by Risk Category
    # -------------------------------
    st.write("🔹 Student Segmentation by Risk")
    st.bar_chart(df_predictions['Risk_Category'].value_counts())
