import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import base64

# Constants
IMG_SIZE = (128, 128)
MODEL_PATH = "fingerprint_bloodgroup_model.h5"
CLASS_NAMES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_blood_group(model, image: Image.Image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)[0]
    class_index = np.argmax(prediction)
    confidence = float(prediction[class_index]) * 100
    return CLASS_NAMES[class_index], confidence

# Background CSS
def add_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Custom title style
def custom_title():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');

        .custom-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 42px;
            text-align: center;
            padding: 20px;
            background: linear-gradient(to right, #000000, #8B0000, #000000);
            color: white;
            border-radius: 15px;
            margin-top: 20px;
            box-shadow: 0 0 25px #8B0000AA;
        }
        </style>

        <div class='custom-title'>
            🔬 Blood Group Detection Through Fingerprint
        </div>
    """, unsafe_allow_html=True)

# Streamlit app config
st.set_page_config("Blood Group Detection Through Fingerprint", layout="centered")
add_background("background.jpg")
custom_title()

st.markdown("### Upload a fingerprint image (BMP, JPG, PNG) to predict the blood group using AI.")

uploaded_file = st.file_uploader("Upload Fingerprint Image", type=["bmp", "jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Fingerprint", use_column_width=True)

    with st.spinner("🔍 Analyzing fingerprint..."):
        model = load_model()
        prediction, confidence = predict_blood_group(model, image)

    st.markdown("## 🧬 Prediction Result")
    st.success(f"**Predicted Blood Group:** {prediction}")
    st.caption(f"Confidence: {confidence:.2f}%")
