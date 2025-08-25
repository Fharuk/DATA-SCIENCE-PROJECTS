# ===============================
# Streamlit App: Without SHAP Waterfall
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ---------------------------
# Load Model & Encoders
# ---------------------------
rf_model = joblib.load("random_forest_student_performance.pkl")
gender_le = joblib.load("label_encoder_gender.pkl")
lunch_le = joblib.load("label_encoder_lunch.pkl")
prep_le = joblib.load("label_encoder_test_preparation_course.pkl")

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.title("🎓 Student Performance Prediction Dashboard")

# ---------------------------
# Sidebar: Student Inputs
# ---------------------------
st.sidebar.header("Enter Student Details")

gender = st.sidebar.selectbox("Gender", ["female", "male"])
lunch = st.sidebar.selectbox("Lunch Type", ["standard", "free/reduced"])
prep_course = st.sidebar.selectbox("Test Preparation Course", ["none", "completed"])
math_score = st.sidebar.slider("Math Score", 0, 100, 70)
reading_score = st.sidebar.slider("Reading Score", 0, 100, 70)
writing_score = st.sidebar.slider("Writing Score", 0, 100, 70)
race_ethnicity = st.sidebar.selectbox("Race/Ethnicity", ["group A","group B","group C","group D","group E"])
parent_edu = st.sidebar.selectbox("Parental Level of Education", [
    "some high school","high school","some college","associate's degree","bachelor's degree","master's degree"
])

# ---------------------------
# Preprocess Input
# ---------------------------
input_df = pd.DataFrame({
    "gender": [gender_le.transform([gender])[0]],
    "lunch": [lunch_le.transform([lunch])[0]],
    "test_preparation_course": [prep_le.transform([prep_course])[0]],
    "math_score": [math_score],
    "reading_score": [reading_score],
    "writing_score": [writing_score],
    "subject_weakness": [int(math_score<50 or reading_score<50 or writing_score<50)]
})

# One-hot encode race_ethnicity
for grp in ["group B","group C","group D","group E"]:
    col_name = f"race_ethnicity_{grp.replace(' ','_')}"
    input_df[col_name] = 1 if race_ethnicity == grp else 0

# One-hot encode parental education
for edu in ["bachelor's degree","high school","master's degree","some college","some high school"]:
    col_name = f'parental_level_of_education_{edu.replace(" ", "_").replace("\'", "")}'
    input_df[col_name] = 1 if parent_edu == edu else 0

# ---------------------------
# Ensure all training columns exist
# ---------------------------
train_columns = rf_model.feature_names_in_
for col in train_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[train_columns]

# ---------------------------
# Predict Performance
# ---------------------------
prediction = rf_model.predict(input_df)[0]
prediction_proba = rf_model.predict_proba(input_df)[0]

# Detect Weak Subjects
weak_subjects = []
if math_score < 50:
    weak_subjects.append("Math")
if reading_score < 50:
    weak_subjects.append("Reading")
if writing_score < 50:
    weak_subjects.append("Writing")

# ---------------------------
# Tabs Layout
# ---------------------------
tab1, tab2 = st.tabs(["Prediction", "Global Insights"])

# --------- Tab 1: Prediction ---------
with tab1:
    st.header("Prediction Results")
    color = "green" if prediction=="High" else "orange" if prediction=="Average" else "red"
    st.markdown(f"<h2 style='color:{color}'>{prediction}</h2>", unsafe_allow_html=True)
    
    st.subheader("Prediction Probabilities")
    proba_df = pd.DataFrame(prediction_proba.reshape(1,-1), columns=rf_model.classes_)
    st.bar_chart(proba_df.T)
    
    st.subheader("Subjects to Improve 💡")
    if weak_subjects:
        st.warning(", ".join(weak_subjects))
    else:
        st.success("No weak subjects identified! ✅")
    
    st.subheader("Download Prediction Report")
    report_df = input_df.copy()
    report_df["Predicted Performance"] = prediction
    report_df["Weak Subjects"] = ", ".join(weak_subjects) if weak_subjects else "None"
    for idx, cls in enumerate(rf_model.classes_):
        report_df[f"Probability_{cls}"] = prediction_proba[idx]
    csv = report_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Report as CSV",
        data=csv,
        file_name='student_prediction_report.csv',
        mime='text/csv'
    )

# --------- Tab 2: Global Insights ---------
with tab2:
    st.header("Global Feature Importance")
    
    explainer = shap.TreeExplainer(rf_model)
    shap_values_all = explainer.shap_values(input_df)

    st.subheader("Bar Plot")
    shap.summary_plot(shap_values_all, input_df, plot_type="bar", class_names=rf_model.classes_, show=False)
    st.pyplot(plt.gcf())
    plt.clf()
    
    st.subheader("Beeswarm Plot")
    shap.summary_plot(shap_values_all, input_df, feature_names=input_df.columns, class_names=rf_model.classes_, show=False)
    st.pyplot(plt.gcf())
    plt.clf()
