# Deployment-Ready Streamlit App with CSV Download
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved models
daily_model = joblib.load("daily_demand_model.pkl")
hourly_model = joblib.load("hourly_demand_model.pkl")

st.title("🚴 Bike Demand Prediction & Allocation System")

# User Inputs
st.sidebar.header("Input Settings")
date_input = st.sidebar.date_input("Select Date")
total_bikes = st.sidebar.number_input("Total Available Bikes", min_value=100, value=1000, step=50)
season = st.sidebar.selectbox("Season (1:Spring,2:Summer,3:Fall,4:Winter)", [1,2,3,4])
yr = st.sidebar.selectbox("Year (0:2011,1:2012)", [0,1])
mnth = date_input.month
weekday = date_input.weekday()
holiday = st.sidebar.selectbox("Holiday", [0,1])
workingday = st.sidebar.selectbox("Working Day", [0,1])
weathersit = st.sidebar.selectbox("Weather (1-4)", [1,2,3,4])
temp = st.sidebar.slider("Temperature (0-1 normalized)", 0.0, 1.0, 0.5)
atemp = st.sidebar.slider("Feels Like Temp (0-1 normalized)", 0.0, 1.0, 0.5)
hum = st.sidebar.slider("Humidity (0-1 normalized)", 0.0, 1.0, 0.5)
windspeed = st.sidebar.slider("Wind Speed (0-1 normalized)", 0.0, 1.0, 0.5)

# Generate hourly features
hours = list(range(24))
hourly_input = pd.DataFrame({
    'season':[season]*24,
    'yr':[yr]*24,
    'mnth':[mnth]*24,
    'hr':hours,
    'holiday':[holiday]*24,
    'weekday':[weekday]*24,
    'workingday':[workingday]*24,
    'weathersit':[weathersit]*24,
    'temp':[temp]*24,
    'atemp':[atemp]*24,
    'hum':[hum]*24,
    'windspeed':[windspeed]*24,
    'hr_sin':np.sin(2*np.pi*np.array(hours)/24),
    'hr_cos':np.cos(2*np.pi*np.array(hours)/24),
    'is_weekend':[1 if weekday>=5 else 0]*24,
    'working_hr':[workingday*hr for hr in hours]
})

# Ensure hourly features match model
hourly_model_features = hourly_model.feature_name()

for f in hourly_model_features:
    if f not in hourly_input.columns:
        hourly_input[f] = 0  # default value

hourly_input = hourly_input[hourly_model_features]


# Predict hourly demand
hourly_input['predicted_cnt'] = hourly_model.predict(hourly_input)

# Prepare daily features
daily_features_input = hourly_input.drop(columns=['hr','hr_sin','hr_cos','working_hr',
                                                  'cnt_lag1','cnt_lag24','cnt_lag168',
                                                  'cnt_roll24','cnt_roll168','predicted_cnt'])

daily_model_features = daily_model.feature_name()
for f in daily_model_features:
    if f not in daily_features_input.columns:
        daily_features_input[f] = 0  # default value

daily_features_input = daily_features_input[daily_model_features]

daily_pred = daily_model.predict(daily_features_input.head(1))[0]

st.subheader("📊 Predicted Daily Bike Demand")
st.metric("Daily Demand", f"{int(daily_pred)} bikes")

# Allocate bikes proportionally
hourly_input['allocated_bikes'] = (
    hourly_input['predicted_cnt']/hourly_input['predicted_cnt'].sum()*total_bikes
).round().astype(int)

# Visualizations
st.subheader("⏱ Hourly Bike Demand Forecast")
st.line_chart(hourly_input.set_index('hr')['predicted_cnt'])

st.subheader("🚲 Hourly Bike Allocation")
st.bar_chart(hourly_input.set_index('hr')['allocated_bikes'])


# Peak hours detection
peak_hours = hourly_input[hourly_input['predicted_cnt'] > hourly_input['predicted_cnt'].quantile(0.9)]
st.subheader("⚡ Peak Demand Hours")
st.table(peak_hours[['hr','predicted_cnt','allocated_bikes']])


# CSV Download
st.subheader("💾 Download Predictions")

csv_data = hourly_input[['hr','predicted_cnt','allocated_bikes']].copy()
csv_data['date'] = str(date_input)

csv = csv_data.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download Hourly Predictions as CSV",
    data=csv,
    file_name=f"bike_predictions_{date_input}.csv",
    mime='text/csv'
)
