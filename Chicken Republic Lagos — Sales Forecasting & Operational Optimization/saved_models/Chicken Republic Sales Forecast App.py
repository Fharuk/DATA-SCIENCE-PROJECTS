# ============================================================
# app.py — Streamlit Sales Forecast App (Updated)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor

st.set_page_config(page_title="Chicken Republic Sales Forecast", layout="wide")

# -------------------------------
# Define feature columns directly
# -------------------------------
feature_cols = [
    "Year", "Month", "Day", "DayOfWeek", "WeekOfYear", "IsWeekend",
    "qty_lag_1", "qty_lag_7", "roll_mean_7", "roll_std_7", "price_delta_7",
    "Location_enc", "Product Category_enc", "Product_enc"
]

# -------------------------------
# Load Latest Sales Data
# -------------------------------
df = pd.read_excel("chicken_republic_lagos_sales (1).xlsx")

# -------------------------------
# Train Quantile Models
# -------------------------------
quantiles = [0.1, 0.5, 0.9]
model_dict = {}
for q in quantiles:
    lgb_q = LGBMRegressor(
        objective='quantile',
        alpha=q,
        learning_rate=0.1,
        num_leaves=31,
        n_estimators=200,
        random_state=42
    )
    # Prepare features for training
    df_feat = df.copy()
    df_feat["Year"] = df_feat["Date"].dt.year
    df_feat["Month"] = df_feat["Date"].dt.month
    df_feat["Day"] = df_feat["Date"].dt.day
    df_feat["DayOfWeek"] = df_feat["Date"].dt.dayofweek
    df_feat["WeekOfYear"] = df_feat["Date"].dt.isocalendar().week
    df_feat["IsWeekend"] = df_feat["DayOfWeek"].isin([5,6]).astype(int)
    df_feat["qty_lag_1"] = df_feat["Quantity Sold"].shift(1).fillna(0)
    df_feat["qty_lag_7"] = df_feat["Quantity Sold"].shift(7).fillna(0)
    df_feat["roll_mean_7"] = df_feat["Quantity Sold"].shift(1).rolling(7, min_periods=1).mean().fillna(0)
    df_feat["roll_std_7"] = df_feat["Quantity Sold"].shift(1).rolling(7, min_periods=1).std().fillna(0)
    df_feat["price_delta_7"] = df_feat["Unit Price (NGN)"] - df_feat["Unit Price (NGN)"].shift(1).rolling(7, min_periods=1).mean().fillna(0)
    
    # Encode categoricals
    for col in ["Location", "Product Category", "Product"]:
        le = LabelEncoder()
        df_feat[col + "_enc"] = le.fit(df[col]).transform(df_feat[col])
    
    X_train = df_feat[feature_cols]
    y_train = df_feat["Quantity Sold"]
    
    lgb_q.fit(X_train, y_train)
    model_dict[q] = lgb_q

# -------------------------------
# Streamlit Sidebar Inputs
# -------------------------------
st.title("Chicken Republic Lagos — Sales Forecasting")
product = st.selectbox("Select Product", sorted(df["Product"].unique()))
location = st.selectbox("Select Location", sorted(df["Location"].unique()))

# -------------------------------
# Filter Data
# -------------------------------
df_sku = df[(df["Product"] == product) & (df["Location"] == location)].sort_values("Date").reset_index(drop=True)

# -------------------------------
# Feature Engineering for Prediction
# -------------------------------
last_row = df_sku.iloc[-1:].copy()
last_row["Year"] = last_row["Date"].dt.year
last_row["Month"] = last_row["Date"].dt.month
last_row["Day"] = last_row["Date"].dt.day
last_row["DayOfWeek"] = last_row["Date"].dt.dayofweek
last_row["WeekOfYear"] = last_row["Date"].dt.isocalendar().week
last_row["IsWeekend"] = last_row["DayOfWeek"].isin([5,6]).astype(int)
last_row["qty_lag_1"] = df_sku["Quantity Sold"].shift(1).iloc[-1]
last_row["qty_lag_7"] = df_sku["Quantity Sold"].shift(7).iloc[-1] if len(df_sku) > 7 else df_sku["Quantity Sold"].iloc[-1]
last_row["roll_mean_7"] = df_sku["Quantity Sold"].shift(1).rolling(7, min_periods=1).mean().iloc[-1]
last_row["roll_std_7"] = df_sku["Quantity Sold"].shift(1).rolling(7, min_periods=1).std().fillna(0).iloc[-1]
last_row["price_delta_7"] = last_row["Unit Price (NGN)"] - df_sku["Unit Price (NGN)"].shift(1).rolling(7, min_periods=1).mean().fillna(0).iloc[-1]

# Encode categoricals
for col in ["Location", "Product Category", "Product"]:
    le = LabelEncoder()
    last_row[col + "_enc"] = le.fit(df[col]).transform(last_row[col])

X_new = last_row[feature_cols]

# -------------------------------
# Predict Quantiles
# -------------------------------
forecast = {f"q{int(q*100)}": model.predict(X_new)[0] for q, model in model_dict.items()}
st.subheader("Forecasted Sales Quantiles (Next Day)")
st.write(forecast)

# -------------------------------
# Historical Sales Plot
# -------------------------------
st.subheader("Historical Sales")
plt.figure(figsize=(10,4))
plt.plot(df_sku["Date"], df_sku["Quantity Sold"], marker='o', label="Actual Sales")
plt.axhline(y=forecast["q50"], color='r', linestyle='--', label="Forecast P50")
plt.fill_between(df_sku["Date"].iloc[-1:], forecast["q10"], forecast["q90"], color='orange', alpha=0.2, label="P10-P90 Range")
plt.xlabel("Date")
plt.ylabel("Quantity Sold")
plt.legend()
st.pyplot(plt)

# -------------------------------
# Download Forecast CSV
# -------------------------------
next_day = df_sku["Date"].max() + pd.Timedelta(days=1)
forecast_df = pd.DataFrame([forecast], index=[next_day])
csv = forecast_df.to_csv(index=True).encode('utf-8')
st.download_button(
    label="Download Forecast CSV",
    data=csv,
    file_name=f"forecast_{product}_{location}.csv",
    mime='text/csv',
)
