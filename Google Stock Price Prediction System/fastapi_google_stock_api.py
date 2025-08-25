# -------------------------------
# fastapi_google_stock_api.py
# -------------------------------

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
import joblib
import json
import io

# -------------------------------
# 1. Load Saved Artifacts
# -------------------------------
model = load_model("google_stock_lstm_model.keras", compile=False)
scaler = joblib.load("scaler_google_stock.save")

with open("config.json", "r") as f:
    config = json.load(f)

sequence_length = config["sequence_length"]
feature_names = config["feature_names"]

# -------------------------------
# 2. Feature Engineering Function
# -------------------------------
def add_features(df):
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['BB_High'] = df['Close'].rolling(20).mean() + 2*df['Close'].rolling(20).std()
    df['BB_Low'] = df['Close'].rolling(20).mean() - 2*df['Close'].rolling(20).std()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Rolling_Volatility_10'] = df['Daily_Return'].rolling(10).std()
    df['Rolling_Volatility_20'] = df['Daily_Return'].rolling(20).std()
    for lag in [1,2,3,5,10]:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
    df = df.dropna()
    return df

# -------------------------------
# 3. Preprocessing Function
# -------------------------------
def prepare_input(new_df):
    df_features = add_features(new_df)
    scaled = scaler.transform(df_features[feature_names])
    X_seq = []
    X_seq.append(scaled[-sequence_length:])
    return np.array(X_seq)

# -------------------------------
# 4. Prediction Function
# -------------------------------
def predict_next_close(new_df):
    X_seq = prepare_input(new_df)
    pred_scaled = model.predict(X_seq)
    close_min = scaler.data_min_[0]
    close_max = scaler.data_max_[0]
    pred_usd = pred_scaled[0][0] * (close_max - close_min) + close_min
    return pred_usd

# -------------------------------
# 5. Initialize FastAPI
# -------------------------------
app = FastAPI(title="Google Stock Price LSTM API", version="1.0")

@app.get("/")
def home():
    return {"message": "Google Stock Price Prediction API is running."}

@app.post("/predict_csv/")
async def predict_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file with raw stock data (Date, Open, High, Low, Close, Volume)
    and get the predicted next closing price.
    """
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    predicted_price = predict_next_close(df)
    return {"predicted_next_close": round(predicted_price, 2)}

@app.post("/predict_json/")
async def predict_json(data: list):
    """
    Send JSON data (list of dicts) with raw stock rows and get prediction.
    Example input: [{"Date":"2025-01-01","Open":100,"High":102,"Low":99,"Close":101,"Volume":5000000}, ...]
    """
    df = pd.DataFrame(data)
    predicted_price = predict_next_close(df)
    return {"predicted_next_close": round(predicted_price, 2)}
