# -------------------------------
# Google_Stock_Prediction: Real-Time Stock Data with Interactive Predictions (Fixed API)
# -------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import json

st.set_page_config(page_title="Google Stock Prediction", layout="wide")
st.title("Google Stock Price Prediction System (Fixed API)")
st.write("Fetch live Google stock data, predict future prices, and visualize historical trends with interactive charts.")

# -------------------------------
# Data Source Selection
# -------------------------------
st.header("Data Source")
data_source = st.radio("Select data source:", ("Real-Time Google Stock", "Upload CSV", "Paste JSON"))

df = None

# -------------------------------
# Real-Time Google Stock via yfinance
# -------------------------------
if data_source == "Real-Time Google Stock":
    ticker = "GOOG"
    period = st.selectbox("Select period for historical data:", ["1y", "2y", "5y", "10y"])
    interval = st.selectbox("Select interval:", ["1d", "1wk", "1mo"])
    
    with st.spinner("Fetching real-time data..."):
        df = yf.download(ticker, period=period, interval=interval)
        df.reset_index(inplace=True)
        st.write("Preview of fetched data:")
        st.dataframe(df.tail())

# -------------------------------
# CSV Upload
# -------------------------------
elif data_source == "Upload CSV":
    csv_file = st.file_uploader("Upload CSV with columns: Date, Open, High, Low, Close, Volume", type=["csv"])
    if csv_file is not None:
        df = pd.read_csv(csv_file)
        st.write("Preview of uploaded CSV:")
        st.dataframe(df.head())

# -------------------------------
# JSON Input
# -------------------------------
elif data_source == "Paste JSON":
    json_text = st.text_area("Enter JSON array with stock data:", height=200)
    if json_text:
        try:
            df_json = pd.DataFrame(json.loads(json_text))
            st.write("Preview of pasted JSON data:")
            st.dataframe(df_json.head())
            df = df_json
        except:
            st.error("Invalid JSON format.")

# -------------------------------
# Prediction & Visualization
# -------------------------------
if df is not None:
    num_days = st.slider("Number of future days to predict", min_value=1, max_value=10, value=3)
    
    if st.button("Predict & Visualize Future Prices"):
        # ---- Prepare Data for API ----
        try:
            df_api = df.copy()
            if 'Date' in df_api.columns:
                df_api['Date'] = df_api['Date'].astype(str)  # Ensure JSON-serializable
            # Keep only numeric columns for prediction
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df_api = df_api[numeric_cols]
            data_json = df_api.to_dict(orient='records')

            # ---- Call FastAPI ----
            response = requests.post("http://127.0.0.1:8000/predict_json/", json=data_json)
            if response.status_code == 200:
                predicted_price = response.json()['predicted_next_close']
                st.success(f"Predicted Next Closing Price: ${predicted_price:.2f}")
            else:
                st.error("Error in prediction. Check API.")
                predicted_price = None
        except Exception as e:
            st.error(f"Error connecting to API: {e}")
            predicted_price = None
        
        # ---- Preprocess and Add Indicators ----
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['BB_High'] = df['Close'].rolling(20).mean() + 2*df['Close'].rolling(20).std()
        df['BB_Low'] = df['Close'].rolling(20).mean() - 2*df['Close'].rolling(20).std()
        
        # ---- Future Predictions ----
        if predicted_price is not None:
            future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=num_days)
            future_prices = [predicted_price * (1 + 0.01*i) for i in range(num_days)]
            lower_bound = [p * 0.98 for p in future_prices]
            upper_bound = [p * 1.02 for p in future_prices]
            df_future = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Close': future_prices,
                'Lower_CI': lower_bound,
                'Upper_CI': upper_bound
            })
        
        # ---- Plotly Interactive Chart ----
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], mode='lines', name='SMA 20', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_20'], mode='lines', name='EMA 20', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_High'], mode='lines', name='BB High', line=dict(color='gray'), opacity=0.5))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Low'], mode='lines', name='BB Low', line=dict(color='gray'), opacity=0.5, fill='tonexty', fillcolor='rgba(128,128,128,0.2)'))
        
        if predicted_price is not None:
            fig.add_trace(go.Scatter(x=df_future['Date'], y=df_future['Predicted_Close'], mode='lines+markers', name='Predicted Future', line=dict(color='red', dash='dash')))
            fig.add_trace(go.Scatter(
                x=pd.concat([df_future['Date'], df_future['Date'][::-1]]),
                y=pd.concat([df_future['Upper_CI'], df_future['Lower_CI'][::-1]]),
                fill='toself', fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,0,0,0)'), name='Confidence Interval'
            ))
        
        fig.update_layout(title=f"Google Stock Prices & Predictions",
                          xaxis_title="Date",
                          yaxis_title="Price (USD)",
                          hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        # ---- Correlation Heatmap ----
        st.subheader("Feature Correlation Heatmap")
        numeric_cols_df = df.select_dtypes(include='number')
        corr = numeric_cols_df.corr()
        fig2 = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig2, use_container_width=True)
        
        # ---- Download Predictions ----
        if predicted_price is not None:
            st.subheader("Download Predicted Future Prices")
            df_future_download = df_future.rename(columns={'Predicted_Close':'Close'})
            csv_data = df_future_download.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv_data, "predicted_future_prices.csv", "text/csv")
