# Predictive_maintenance.py

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load trained model
model = joblib.load("maintenance_for_ev_range_predictor.pkl")

# App title
st.title("Predictive Maintenance for EVs")
st.write("""
Predict potential maintenance needs for electric vehicles based on key specifications.
You can enter EV data manually or upload a CSV file with multiple EVs.
""")

# Sidebar for threshold
threshold_factor = st.sidebar.slider("Alert Threshold Factor", 0.5, 1.0, 0.90)

# Features used during model training
trained_features = [
    'battery_capacity_kWh',
    'efficiency_wh_per_km',
    'top_speed_kmh',
    'acceleration_0_100_s',
    'fast_charge_minutes',
    'torque_per_cell',
    'performance_index'
]

# Function to compute engineered features
def compute_engineered_features(df):
    df = df.copy()
    # Compute torque per cell
    df['torque_per_cell'] = df['torque_nm'] / df['number_of_cells'].replace(0, 1)
    # Compute performance index
    df['performance_index'] = (df['range_km'] / df['battery_capacity_kWh'].replace(0,1)) * df['torque_per_cell'] / df['acceleration_0_100_s'].replace(0,1)
    return df

# Show feature importances
st.subheader("Feature Importances")
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': trained_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
st.dataframe(importance_df)

fig = px.bar(importance_df,
             x='Importance',
             y='Feature',
             orientation='h',
             color='Importance',
             color_continuous_scale='Viridis',
             title='Feature Importances for EV Maintenance Prediction')
st.plotly_chart(fig)

# Input method selection
input_method = st.radio("Select Input Method", ["Manual Entry", "Upload CSV"])

if input_method == "Manual Entry":
    st.sidebar.header("Enter EV Specifications")
    battery_capacity = st.sidebar.number_input("Battery Capacity (kWh)", 10.0, 150.0, 75.0)
    efficiency = st.sidebar.number_input("Efficiency (Wh/km)", 100.0, 500.0, 180.0)
    top_speed = st.sidebar.number_input("Top Speed (km/h)", 100, 350, 180)
    acceleration = st.sidebar.number_input("Acceleration 0-100 km/h (s)", 2.0, 20.0, 7.0)
    fast_charge = st.sidebar.number_input("Fast Charge Time (minutes)", 0.0, 200.0, 60.0)
    torque = st.sidebar.number_input("Torque (Nm)", 50, 1500, 300)
    number_of_cells = st.sidebar.number_input("Number of Cells", 50, 5000, 200)
    actual_range = st.sidebar.number_input("Actual Range (km)", 50, 1000, 400)

    # Create input DataFrame
    input_data = pd.DataFrame({
        'battery_capacity_kWh': [battery_capacity],
        'efficiency_wh_per_km': [efficiency],
        'top_speed_kmh': [top_speed],
        'acceleration_0_100_s': [acceleration],
        'fast_charge_minutes': [fast_charge],
        'torque_nm': [torque],
        'number_of_cells': [number_of_cells],
        'range_km': [actual_range]  # needed for engineered features
    })

    # Compute engineered features
    input_data = compute_engineered_features(input_data)

    # Keep only trained features
    input_data_model = input_data[trained_features]

    # Predict
    predicted_range = model.predict(input_data_model)[0]
    maintenance_alert = predicted_range < (threshold_factor * actual_range)

    st.subheader("Prediction Results")
    st.write(f"Predicted Range: {predicted_range:.2f} km")
    st.write(f"Actual Range: {actual_range} km")
    if maintenance_alert:
        st.error("⚠️ Maintenance Alert: This EV may require attention!")
    else:
        st.success("✅ EV is operating within expected range limits.")

else:
    st.subheader("Upload CSV for Batch Predictions")
    st.write("CSV must contain columns: 'battery_capacity_kWh','efficiency_wh_per_km','top_speed_kmh','acceleration_0_100_s','fast_charge_minutes','torque_nm','number_of_cells','range_km','actual_range_km'")

    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(df)

        # Compute engineered features
        df = compute_engineered_features(df)

        # Keep only trained features
        input_features = df[trained_features]

        # Predict ranges
        predicted_ranges = model.predict(input_features)
        df['predicted_range_km'] = predicted_ranges
        df['maintenance_alert'] = df['predicted_range_km'] < (threshold_factor * df['actual_range_km'])

        # Show alerts table
        alerts_df = df[df['maintenance_alert'] == True]
        st.subheader("EVs Flagged for Maintenance")
        st.dataframe(alerts_df)

        # Visualize flagged EVs
        if not alerts_df.empty:
            fig_alert = px.bar(alerts_df.sort_values(by='predicted_range_km'),
                               x='predicted_range_km',
                               y=alerts_df.index,
                               color='predicted_range_km',
                               color_continuous_scale='Reds',
                               labels={'y':'EV Index'},
                               title=f'EVs Flagged for Maintenance (Threshold: {threshold_factor*100:.0f}%)')
            st.plotly_chart(fig_alert)
        else:
            st.info("No EVs flagged for maintenance based on the current threshold.")
