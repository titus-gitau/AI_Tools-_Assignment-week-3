# ================================================================
# MNIST CLASSIFIER WEB APP USING STREAMLIT
# ================================================================
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ------------------------------------------------
# Load trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mnist_cnn_model.h5")
    return model

model = load_model()

# ------------------------------------------------
# App title and description
st.title("🧠 MNIST Handwritten Digit Classifier")
st.write("Upload or draw a digit (0–9) to see what the model predicts.")

# ------------------------------------------------
# File uploader
uploaded_file = st.file_uploader("Upload a digit image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')    # Convert to grayscale
    image = image.resize((28, 28))                    # Resize to 28×28
    img_array = np.array(image) / 255.0               # Normalize to [0, 1]
    img_array = img_array.reshape(1, 28, 28, 1)       # Reshape for CNN

    # Predict
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Show image and prediction
    st.image(image, caption=f"Predicted: {predicted_label} ({confidence:.2f}% confidence)", width=150)
    st.bar_chart(prediction[0])
else:
    st.info("👆 Upload a digit image to get a prediction.")

st.caption("Built with TensorFlow + Streamlit")
