import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="wide"
)

# Professional Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: 800;
        color: #4B6BFF;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 18px;
        color: #A0A0A0;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        background-color: #4B6BFF;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<p class="main-header">🔢 Handwritten Digit Recognition</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Deep Learning CNN Model Powered by TensorFlow & Streamlit</p>', unsafe_allow_html=True)

# Load Model directly from local directory (mnist_model.h5)
@st.cache_resource
def load_model():
    model_path = 'mnist_model.h5'
    return tf.keras.models.load_model(model_path)

with st.spinner("Loading AI model... Please wait."):
    model = load_model()

# Sidebar Info
with st.sidebar:
    st.title("📊 Project Dashboard")
    st.info("This application predicts handwritten digits (0 to 9) with high accuracy using a Convolutional Neural Network (CNN).")
    st.markdown("---")
    st.markdown("### 🎯 Model Overview")
    st.write("- **Architecture:** Convolutional Neural Network (CNN)")
    st.write("- **Dataset:** MNIST Dataset")
    st.write("- **Target Accuracy:** ~99%")
    st.write("- **Framework:** TensorFlow / Keras")

# Main Layout
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 📤 Upload Digit Image")
    st.markdown("Please upload a clear image of a handwritten digit (0-9):")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Digit Image", width=280)

with col2:
    st.markdown("### 🔍 Prediction & Analytics")
    if uploaded_file is not None:
        with st.spinner("Analyzing digit..."):
            processed_img = image.convert('L').resize((28, 28))
            img_array = np.array(processed_img) / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)
            
            preds = model.predict(img_array)
            pred_idx = np.argmax(preds[0])
            confidence = float(np.max(preds[0])) * 100
        
        # Display Results
        st.success(f"### Predicted Digit: **{pred_idx}**")
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
        st.progress(int(confidence))
        
        st.markdown("---")
        st.info("💡 **Tip:** For best accuracy, use dark writing on a clean white background.")
    else:
        st.info("👈 Please upload a digit image from the left panel to see predictions.")
