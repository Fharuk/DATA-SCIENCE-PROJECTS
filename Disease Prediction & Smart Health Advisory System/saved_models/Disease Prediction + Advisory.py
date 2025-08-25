# ============================================
# Streamlit Disease Prediction + Advisory App
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import shap

# ----------------------------
# 1. Load Models & Encoder
# ----------------------------
MODEL_DIR = r"C:\Users\AKINMADE FARUQ\Downloads\PROJECT MATERIALS\My Projects\New upload\Disease Prediction\saved_models"

rf_model_path = os.path.join(MODEL_DIR, "random_forest_model.pkl")
label_encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
explainer_path = os.path.join(MODEL_DIR, "rf_shap_explainer.pkl")

rf_model = joblib.load(rf_model_path)
label_encoder = joblib.load(label_encoder_path)
explainer = joblib.load(explainer_path)

# ----------------------------
# 2. Advisory Dictionary
# ----------------------------
advisory_dict = {
    "(vertigo) Paroymsal  Positional Vertigo": "Sit or lie down if dizzy. Consult a neurologist if episodes persist.",
    "AIDS": "Immediate consultation with an infectious disease specialist is required. Follow prescribed ART therapy.",
    "Acne": "Maintain proper hygiene, use dermatologist-recommended topical treatments, avoid excessive oil-based products.",
    "Alcoholic hepatitis": "Stop alcohol consumption immediately. Seek hepatologist consultation and follow treatment plan.",
    "Allergy": "Identify and avoid allergens. Consider antihistamines and consult an allergist if severe.",
    "Arthritis": "Maintain joint-friendly activity. Consult a rheumatologist for proper management and medication.",
    "Bronchial Asthma": "Carry prescribed inhalers. Avoid triggers and consult a pulmonologist for management.",
    "Cervical spondylosis": "Maintain proper posture, do neck exercises, consult an orthopedic specialist if pain persists.",
    "Chicken pox": "Isolate to prevent spread, manage fever and itching, consult a doctor if complications arise.",
    "Chronic cholestasis": "Consult a hepatologist for liver function management and follow dietary recommendations.",
    "Common Cold": "Rest, hydrate, and use OTC remedies. See a doctor if fever or symptoms worsen.",
    "Dengue": "Seek immediate medical care. Monitor platelet count and follow physician guidance.",
    "Diabetes": "Regularly monitor blood sugar, follow dietary plan, and consult an endocrinologist.",
    "Dimorphic hemmorhoids(piles)": "Maintain proper hygiene, high-fiber diet, and consult a proctologist if bleeding persists.",
    "Drug Reaction": "Stop the suspected medication and consult a physician immediately.",
    "Fungal infection": "Keep affected area clean and dry. Consult a dermatologist for antifungal treatment.",
    "GERD": "Avoid trigger foods, eat smaller meals, and consult a gastroenterologist for persistent symptoms.",
    "Gastroenteritis": "Stay hydrated, follow light diet, and consult a doctor if symptoms are severe.",
    "Heart attack": "Seek emergency medical attention immediately. Do not ignore chest pain.",
    "Hepatitis A": "Consult a hepatologist, maintain hydration, and follow a liver-friendly diet.",
    "Hepatitis B": "Consult a hepatologist and follow prescribed antiviral therapy.",
    "Hepatitis C": "Consult a hepatologist for antiviral therapy and monitor liver function.",
    "Hepatitis D": "Immediate hepatology consultation is required for proper management.",
    "Hepatitis E": "Maintain hydration and consult a doctor for supportive care.",
    "Hypertension": "Monitor blood pressure, maintain a low-salt diet, and follow physician guidance.",
    "Hyperthyroidism": "Consult an endocrinologist for medication or therapy management.",
    "Hypoglycemia": "Consume quick-acting glucose when symptomatic and monitor blood sugar regularly.",
    "Hypothyroidism": "Follow prescribed thyroid hormone therapy and monitor levels regularly.",
    "Impetigo": "Maintain hygiene and consult a dermatologist for antibiotic treatment.",
    "Jaundice": "Consult a hepatologist, avoid alcohol, and follow liver-friendly diet recommendations.",
    "Malaria": "Seek immediate medical attention. Follow prescribed antimalarial therapy and hydrate well.",
    "Migraine": "Identify triggers, manage stress, and consult a neurologist for treatment options.",
    "Osteoarthristis": "Maintain joint-friendly activity, consider physiotherapy, and consult a rheumatologist.",
    "Paralysis (brain hemorrhage)": "Seek emergency medical care immediately.",
    "Peptic ulcer diseae": "Avoid NSAIDs, follow physician dietary guidance, and consider proton pump inhibitors.",
    "Pneumonia": "Seek medical care, complete prescribed antibiotics, and maintain hydration.",
    "Psoriasis": "Consult a dermatologist for topical or systemic treatments.",
    "Tuberculosis": "Seek immediate medical care and follow the full course of anti-TB therapy.",
    "Typhoid": "Consult a doctor, complete prescribed antibiotics, and maintain hydration.",
    "Urinary tract infection": "Consult a physician, drink plenty of fluids, and complete prescribed antibiotics.",
    "Varicose veins": "Elevate legs, wear compression stockings, and consult a vascular specialist if needed."
}

# ----------------------------
# 3. App Title
# ----------------------------
st.title("🩺 Smart Disease Prediction & Advisory System")

# ----------------------------
# 4. Single Patient Symptom Input
# ----------------------------
st.sidebar.header("Select Symptoms for Single Patient")
symptom_features = rf_model.feature_names_in_

user_input = {symptom: st.sidebar.checkbox(symptom, value=False) for symptom in symptom_features}
input_df = pd.DataFrame([user_input])

if st.button("Predict Disease for Single Patient"):
    prediction_encoded = rf_model.predict(input_df)[0]
    prediction_proba = rf_model.predict_proba(input_df)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

    st.subheader("Prediction Result")
    st.write(f"**Most likely disease:** {prediction_label}")

    top_indices = np.argsort(prediction_proba)[::-1][:3]
    top_diseases = label_encoder.inverse_transform(top_indices)
    top_probs = prediction_proba[top_indices]

    st.subheader("Top 3 Probable Diseases")
    for disease, prob in zip(top_diseases, top_probs):
        st.write(f"{disease}: {prob*100:.2f}%")

    # SHAP explanation
    st.subheader("Feature Impact (SHAP Values)")
    shap_values_input = explainer.shap_values(input_df)
    plt.figure()
    shap.summary_plot(shap_values_input, input_df, plot_type="bar", show=False)
    st.pyplot(plt.gcf())

    # Probability bar chart
    st.subheader("Prediction Probability Distribution")
    prob_df = pd.DataFrame({
        "Disease": label_encoder.inverse_transform(np.arange(len(prediction_proba))),
        "Probability": prediction_proba
    }).sort_values(by="Probability", ascending=False)
    st.bar_chart(prob_df.set_index("Disease").head(10))

    # Health Advisory
    st.subheader("💡 Health Advisory")
    st.success(advisory_dict.get(prediction_label, "No advisory available. Consult a healthcare professional."))

# ----------------------------
# 5. Bulk Prediction via CSV
# ----------------------------
st.sidebar.subheader("Or Upload CSV for Bulk Prediction")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        bulk_df = pd.read_csv(uploaded_file)
        missing_cols = [col for col in symptom_features if col not in bulk_df.columns]
        if missing_cols:
            st.error(f"Missing required symptom columns: {missing_cols}")
        else:
            # ONLY keep features the model expects
            X_bulk = bulk_df[symptom_features]

            predictions_encoded = rf_model.predict(X_bulk)
            predictions_labels = label_encoder.inverse_transform(predictions_encoded)
            bulk_df["Predicted Disease"] = predictions_labels

            # Top 3 probabilities
            probs = rf_model.predict_proba(X_bulk)
            top3_list = []
            for prob in probs:
                top_indices = np.argsort(prob)[::-1][:3]
                top_diseases = label_encoder.inverse_transform(top_indices)
                top_probs = prob[top_indices]
                top3_list.append(", ".join([f"{d}:{p*100:.1f}%" for d, p in zip(top_diseases, top_probs)]))
            bulk_df["Top 3 Probable Diseases"] = top3_list

            st.subheader("Bulk Prediction Results")
            st.dataframe(bulk_df)

            # Probability bar charts per patient
            st.subheader("Top Probabilities per Patient")
            for idx, row in bulk_df.iterrows():
                st.markdown(f"**Patient {idx+1} - Predicted: {row['Predicted Disease']}**")
                prob = probs[idx]
                prob_df = pd.DataFrame({
                    "Disease": label_encoder.inverse_transform(np.arange(len(prob))),
                    "Probability": prob
                }).sort_values(by="Probability", ascending=False)
                st.bar_chart(prob_df.set_index("Disease").head(5))  # top 5

            # Download CSV
            csv = bulk_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="bulk_predictions.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Error processing file: {e}")
