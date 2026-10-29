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
st.markdown('<p class="sub-header">Deep Learning CNN Model Powered by Streamlit</p>', unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mnist_model.h5')

with st.spinner("Loading AI model... Please wait."):
    model = load_model()

# Sidebar Info
with st.sidebar:
    st.title("📊 Project Dashboard")
    st.info(" Yeh application 0 se 9 tak ke numbers ko pehchanti hai joConvolutional Neural Network (CNN) par mabni hai.")
    st.markdown("---")
    st.markdown("### 🎯 Accuracy Note")
    st.write("Is dataset ki accuracy aam taur par **98% se 99%** tak hoti hai!")

# Main Two-Column Layout
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 📤 Upload Digit Image")
    uploaded_file = st.file_uploader("Choose a digit image (0-9)...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Digit Image", use_column_width=True)

with col2:
    st.markdown("### 🔍 Prediction & Analytics")
    if uploaded_file is not None:
        with st.spinner("Analyzing digit..."):
            processed_img = image.convert('L').resize((28, 28))
            img_array = np.array(processed_img) / 255.0
            
            # Agar background white aur digit black ho, to invert karne ki zaroorat par sakti hai
            # Lekin standard images ke liye direct reshape:
            img_array = img_array.reshape(1, 28, 28, 1)
            
            preds = model.predict(img_array)
            pred_idx = np.argmax(preds[0])
            confidence = float(np.max(preds[0])) * 100
        
        # Display Results
        st.success(f"### Predicted Digit: **{pred_idx}**")
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
        st.progress(int(confidence))
        
        st.markdown("---")
        st.info("💡 **Tip:** Kisi bhi white paper par pen se saaf saaf number likh kar tasveer upload karein!")
    else:
        st.info("👈 Please upload a digit image on the left panel.")
