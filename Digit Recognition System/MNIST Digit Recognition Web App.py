# ================================================
# Streamlit App with Live Drawing Canvas
# Save this as app.py and run with: streamlit run app.py
# ================================================

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# Load trained model
model = tf.keras.models.load_model("cnn_digit_model_full.keras")

st.title("MNIST Digit Recognition Web App")
st.write("Draw a digit on the canvas or upload an image, and the model will predict it.")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="#000000",  # Black background
    stroke_width=12,
    stroke_color="#FFFFFF",  # White strokes
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Upload option
uploaded_file = st.file_uploader("Or upload an image...", type=["png", "jpg", "jpeg"])

# Select the input image
if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
else:
    if canvas_result.image_data is not None:
        # Convert canvas drawing to PIL image
        image = Image.fromarray(np.uint8(canvas_result.image_data)).convert("L")
    else:
        image = None

if image:
    st.image(image, caption="Input Image", use_column_width=True)
    
    # Preprocess: invert, resize, normalize
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image_array = np.array(image).astype("float32") / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    
    # Predict
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    st.write(f"Predicted Digit: {predicted_class}")
