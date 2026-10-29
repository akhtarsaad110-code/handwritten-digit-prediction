import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

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
st.markdown('<p class="sub-header">Deep Learning CNN Model with Live Drawing Canvas</p>', unsafe_allow_html=True)

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
    st.info("Aap yahan ya toh tasveer upload kar sakte hain ya phir seedha canvas par digit draw kar sakte hain!")
    st.markdown("---")
    st.markdown("### 🎯 Accuracy Note")
    st.write("Is model ki accuracy aam taur par **98% se 99%** hoti hai!")

# Tabs for Upload vs Draw
tab1, tab2 = st.tabs(["✍️ Draw Digit on Canvas", "📤 Upload Image"])

with tab1:
    st.markdown("### Draw a digit (0-9) below:")
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    # Predict Button to safely trigger canvas processing
    if st.button("Predict Digit"):
        if canvas_result is not None and canvas_result.image_data is not None:
            img_data = canvas_result.image_data
            if img_data.size > 0:
                img_data_uint8 = img_data.astype('uint8')
                img = Image.fromarray(img_data_uint8).convert('L')
                
                processed_img = img.resize((28, 28))
                img_array = np.array(processed_img) / 255.0
                img_array = img_array.reshape(1, 28, 28, 1)
                
                preds = model.predict(img_array)
                pred_idx = np.argmax(preds[0])
                confidence = float(np.max(preds[0])) * 100
                
                st.markdown("---")
                st.success(f"### Predicted Digit: **{pred_idx}**")
                st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
                st.progress(int(confidence))
        else:
            st.warning("Pehle canvas par koi digit draw karein!")

with tab2:
    st.markdown("### Upload Digit Image")
    uploaded_file = st.file_uploader("Choose a digit image (0-9)...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Digit Image", width=250)
        
        processed_img = image.convert('L').resize((28, 28))
        img_array = np.array(processed_img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        
        preds = model.predict(img_array)
        pred_idx = np.argmax(preds[0])
        confidence = float(np.max(preds[0])) * 100
        
        st.success(f"### Predicted Digit: **{pred_idx}**")
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
        st.progress(int(confidence))
