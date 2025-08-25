# Crop Recommendation app.py
# ----------------------------------------
# Crop Recommendation System (Streamlit App)
# ----------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

# -----------------------------
# Load trained model and encoder
# -----------------------------
model = joblib.load("random_forest_crop_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# -----------------------------
# Crop Emoji Mapping
# -----------------------------
crop_emojis = {
    "apple": "🍎", "banana": "🍌", "blackgram": "🌱", "chickpea": "🫘",
    "coconut": "🥥", "coffee": "☕", "cotton": "🧵", "grapes": "🍇",
    "jute": "📦", "kidneybeans": "🫘", "lentil": "🥣", "maize": "🌽",
    "mango": "🥭", "mothbeans": "🫘", "mungbean": "🟢", "muskmelon": "🍈",
    "orange": "🍊", "papaya": "🍐", "pigeonpeas": "🟡", "pomegranate": "🍎",
    "rice": "🍚", "watermelon": "🍉"
}

# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="🌱 Crop Recommendation System", layout="wide")

st.title("🌱 Intelligent Crop Recommendation System")
st.markdown("Provide soil and weather conditions to get the best crop suggestions.")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Input Parameters")

N = st.sidebar.slider("Nitrogen (N)", 0, 200, 50)
P = st.sidebar.slider("Phosphorus (P)", 0, 200, 50)
K = st.sidebar.slider("Potassium (K)", 0, 200, 50)
temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 60.0)
ph = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 300.0, 100.0)

# Collect input
input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

# -----------------------------
# Gauge Charts for Inputs
# -----------------------------
st.subheader("📊 Input Conditions Overview")

col1, col2, col3 = st.columns(3)

def plot_gauge(value, title, min_val, max_val):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title},
        gauge={"axis": {"range": [min_val, max_val]}}
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
    return fig

with col1:
    st.plotly_chart(plot_gauge(N, "Nitrogen (N)", 0, 200), use_container_width=True)
    st.plotly_chart(plot_gauge(temperature, "Temperature (°C)", 0, 50), use_container_width=True)

with col2:
    st.plotly_chart(plot_gauge(P, "Phosphorus (P)", 0, 200), use_container_width=True)
    st.plotly_chart(plot_gauge(humidity, "Humidity (%)", 0, 100), use_container_width=True)

with col3:
    st.plotly_chart(plot_gauge(K, "Potassium (K)", 0, 200), use_container_width=True)
    st.plotly_chart(plot_gauge(ph, "Soil pH", 0, 14), use_container_width=True)

# -----------------------------
# Predict Crop Probabilities
# -----------------------------
if st.sidebar.button("🌾 Recommend Crops"):
    st.subheader("🌾 Recommended Crops")

    probabilities = model.predict_proba(input_data)[0]
    crops = label_encoder.classes_

    prob_df = pd.DataFrame({
        "Crop": crops,
        "Confidence (%)": probabilities * 100
    }).sort_values(by="Confidence (%)", ascending=False)

    # Add emojis to crop labels
    prob_df["Crop"] = prob_df["Crop"].apply(lambda c: f"{crop_emojis.get(c, '')} {c.capitalize()}")

    # Top 3 recommendations
    st.success("✅ Top 3 Recommended Crops:")
    top3 = prob_df.head(3)
    st.table(top3.reset_index(drop=True))

    # Highlight Best Crop Probability (Gauge)
    best_crop = top3.iloc[0]["Crop"]
    best_prob = top3.iloc[0]["Confidence (%)"]

    st.subheader("🌟 Best Crop Confidence")
    figGauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=best_prob,
        title={"text": f"Best Crop: {best_crop}"},
        gauge={"axis": {"range": [0, 100]}},
        delta={"reference": 50}
    ))
    st.plotly_chart(figGauge, use_container_width=True)

    # Full Probability Table
    st.subheader("📋 Full Crop Probabilities")
    st.dataframe(prob_df.reset_index(drop=True), use_container_width=True)

    # Probability Bar Chart
    st.subheader("📊 Probability Distribution Across Crops")

    colors = []
    for crop, p in zip(prob_df["Crop"], prob_df["Confidence (%)"]):
        if crop == best_crop:
            colors.append("blue")
        elif p > 70:
            colors.append("green")
        elif p >= 40:
            colors.append("orange")
        else:
            colors.append("red")

    fig_all = go.Figure(go.Bar(
        x=prob_df["Crop"],
        y=prob_df["Confidence (%)"],
        text=[f"{p:.2f}%" for p in prob_df["Confidence (%)"]],
        textposition='auto',
        marker_color=colors
    ))
    fig_all.update_layout(
        xaxis_title="Crop",
        yaxis_title="Confidence (%)",
        yaxis=dict(range=[0, 100]),
        height=650
    )
    st.plotly_chart(fig_all, use_container_width=True)

    # -----------------------------
    # Download Buttons
    # -----------------------------
    st.subheader("💾 Download Recommendations")

    # CSV download
    csv = prob_df.to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name='crop_recommendations.csv',
        mime='text/csv'
    )

    # PDF download (optional, requires reportlab)
    try:
        from io import BytesIO
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        c.setFont("Helvetica", 12)
        c.drawString(30, 750, "Crop Recommendation Report")
        y = 720
        for idx, row in prob_df.iterrows():
            line = f"{row['Crop']}: {row['Confidence (%)']:.2f}%"
            c.drawString(30, y, line)
            y -= 20
        c.save()
        buffer.seek(0)
        st.download_button(
            label="Download as PDF",
            data=buffer,
            file_name='crop_recommendations.pdf',
            mime='application/pdf'
        )
    except ImportError:
        st.info("Install 'reportlab' to enable PDF download: pip install reportlab")
