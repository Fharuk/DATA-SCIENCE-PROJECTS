# ============================================================
# 🌦 Real-Time Weather Prediction API using FastAPI
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import joblib

# ---------------------------
# Load trained pipeline and LabelEncoder
# ---------------------------
pipeline = joblib.load("weather_prediction_pipeline.pkl")
le = joblib.load("weather_label_encoder.pkl")

# ---------------------------
# Initialize FastAPI app
# ---------------------------
app = FastAPI(title="Weather Condition Prediction API", version="1.0")

# ---------------------------
# Request models
# ---------------------------
class WeatherObservation(BaseModel):
    Temperature_C: float
    Apparent_Temperature_C: float
    Humidity: float
    Wind_Speed_kmh: float
    Wind_Bearing_degrees: float
    Visibility_km: float
    Pressure_millibars: float
    Year: int
    Month: int
    Day: int
    Hour: int
    DayOfWeek: int
    Precip_Type: Optional[str] = "rain"
    Season: Optional[str] = "Summer"

class WeatherBatch(BaseModel):
    observations: List[WeatherObservation]

# ---------------------------
# Helper function to prepare data
# ---------------------------
def prepare_data_for_api(obs: WeatherObservation):
    feature_columns = pipeline.named_steps['classifier'].feature_names_in_
    obs_dict = dict.fromkeys(feature_columns, 0)
    
    # Map numeric columns
    mapping = {
        'Temperature_C':'Temperature (C)',
        'Apparent_Temperature_C':'Apparent Temperature (C)',
        'Humidity':'Humidity',
        'Wind_Speed_kmh':'Wind Speed (km/h)',
        'Wind_Bearing_degrees':'Wind Bearing (degrees)',
        'Visibility_km':'Visibility (km)',
        'Pressure_millibars':'Pressure (millibars)',
        'Year':'Year','Month':'Month','Day':'Day','Hour':'Hour','DayOfWeek':'DayOfWeek'
    }
    
    for key, col_name in mapping.items():
        obs_dict[col_name] = getattr(obs, key)
    
    # One-hot encode categorical
    precip_col = f"Precip Type_{obs.Precip_Type}"
    season_col = f"Season_{obs.Season}"
    if precip_col in feature_columns:
        obs_dict[precip_col] = 1
    if season_col in feature_columns:
        obs_dict[season_col] = 1
    
    return obs_dict

# ---------------------------
# API endpoint for batch predictions
# ---------------------------
@app.post("/predict_weather")
def predict_weather(batch: WeatherBatch):
    prepared_list = [prepare_data_for_api(obs) for obs in batch.observations]
    new_data_df = pd.DataFrame(prepared_list)
    
    # Predict and decode
    y_pred_enc = pipeline.predict(new_data_df)
    y_pred = le.inverse_transform(y_pred_enc)
    
    return {"predictions": list(y_pred)}

# ============================================================
# Notes:
# Run the API in terminal:
# uvicorn weather_api:app --reload
# ============================================================
