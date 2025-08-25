import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("car_price_model_rf.pkl")

st.title("🚗 Car Price Prediction App")
st.write("Enter car details below to predict its price (in lakhs).")

# --- User Inputs ---
year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, value=2015)
km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50000, step=1000)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner_type = st.selectbox("Owner Type", ["First", "Second", "Third", "Fourth & Above"])
mileage = st.number_input("Mileage (kmpl or km/kg)", min_value=5.0, max_value=40.0, value=18.0, step=0.1)
engine_cc = st.number_input("Engine CC", min_value=600, max_value=5000, value=1500, step=50)
power_bhp = st.number_input("Power (BHP)", min_value=20.0, max_value=600.0, value=100.0, step=1.0)
seats = st.number_input("Seats", min_value=2, max_value=10, value=5)
tax = st.number_input("Tax (in %)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)

# --- Derived Features (matching training pipeline) ---
car_age = 2025 - year

# Create dataframe for prediction
input_data = pd.DataFrame({
    "Year": [year],
    "Kilometers_Driven": [km_driven],
    "Fuel_Type": [fuel_type],
    "Transmission": [transmission],
    "Owner_Type": [owner_type],
    "Mileage_num": [mileage],
    "Engine_num": [engine_cc],
    "Power_num": [power_bhp],
    "Seats": [seats],
    "Tax": [tax],
    "Car_Age": [car_age]
})

# Prediction
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Estimated Price: {prediction:.2f} lakhs")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
