import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="DigitMind AI - Handwritten Digit Recognition",
    page_icon="🧠",
    layout="wide"
)

# Custom Cyberpunk / Modern Dark UI Styling
st.markdown("""
    <style>
    /* Dark Theme Background with subtle grid lines */
    .stApp {
        background-color: #0b0f19;
        background-image: 
            linear-gradient(rgba(0, 240, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 240, 255, 0.03) 1px, transparent 1px);
        background-size: 40px 40px;
        color: #e0e6ed;
    }

    /* Top Badge */
    .badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        border: 1px solid #00f0ff;
        color: #00f0ff;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.5px;
        background: rgba(0, 240, 255, 0.05);
        margin-bottom: 15px;
    }

    /* Glowing Hero Section Header */
    .hero-title {
        font-size: 52px;
        font-weight: 900;
        color: #ffffff;
        line-height: 1.1;
        margin-bottom: 10px;
    }
    .hero-highlight {
        background: linear-gradient(90deg, #00f0ff, #7000ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        font-size: 18px;
        color: #94a3b8;
        margin-bottom: 30px;
    }

    /* Cyberpunk Styled Card Container */
    .cyber-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(0, 240, 255, 0.2);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.5), inset 0 0 15px rgba(0, 240, 255, 0.05);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }

    /* Math Formula Cards */
    .math-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .math-title {
        color: #00f0ff;
        font-weight: 700;
        font-size: 14px;
    }
    .math-expr {
        font-family: monospace;
        color: #38bdf8;
        font-size: 16px;
        margin: 5px 0;
    }

    /* Hide default Streamlit Header & Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Load Model directly from local directory
@st.cache_resource
def load_model():
    model_path = 'mnist_model.h5'
    return tf.keras.models.load_model(model_path)

with st.spinner("Initializing Deep Learning Engine..."):
    model = load_model()

# Top Hero Section
st.markdown('<div class="badge">✨ DEEP LEARNING • COMPUTER VISION</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">From Handwriting<br>to <span class="hero-highlight">Intelligence.</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Watch a Convolutional Neural Network recognize handwritten digits in real time.</div>', unsafe_allow_html=True)

st.markdown("---")

# Main Section Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
        <div class="cyber-card">
            <h3 style="color: #00f0ff; margin-top:0;">📤 Input Workspace</h3>
            <p style="color: #94a3b8; font-size: 14px;">Upload a digit image to feed into the neural pipeline.</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose image...", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Sample", width=260)

with col2:
    st.markdown("""
        <div class="cyber-card">
            <h3 style="color: #7000ff; margin-top:0;">⚡ Model Prediction</h3>
            <p style="color: #94a3b8; font-size: 14px;">Real-time classification powered by CNN.</p>
        </div>
    """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        with st.spinner("Processing tensors..."):
            processed_img = image.convert('L').resize((28, 28))
            img_array = np.array(processed_img) / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)
            
            preds = model.predict(img_array)
            pred_idx = np.argmax(preds[0])
            confidence = float(np.max(preds[0])) * 100
        
        st.markdown(f"""
            <div style="background: rgba(0, 240, 255, 0.1); border: 1px solid #00f0ff; padding: 20px; border-radius: 12px; text-align: center;">
                <h1 style="color: #00f0ff; font-size: 64px; margin:0;">{pred_idx}</h1>
                <p style="color: #94a3b8; margin:0;">Predicted Digit</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.metric(label="Model Confidence Score", value=f"{confidence:.2f}%")
        st.progress(int(confidence))
    else:
        st.info("👈 Upload an image on the left panel to test the model.")

# Technical Mathematics Section
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("### 🧮 The Mathematics Behind the Magic")

mcol1, mcol2, mcol3 = st.columns(3)

with mcol1:
    st.markdown("""
        <div class="math-card">
            <div class="math-title">Image Input</div>
            <div class="math-expr">28 × 28 × 1</div>
            <div style="color: #94a3b8; font-size: 12px;">Grayscale pixel grid representing handwritten strokes.</div>
        </div>
    """, unsafe_allow_html=True)

with mcol2:
    st.markdown("""
        <div class="math-card">
            <div class="math-title">Convolution & ReLU</div>
            <div class="math-expr">f(x) = max(0, x)</div>
            <div style="color: #94a3b8; font-size: 12px;">Feature maps extract edges, curves, and spatial loops.</div>
        </div>
    """, unsafe_allow_html=True)

with mcol3:
    st.markdown("""
        <div class="math-card">
            <div class="math-title">Softmax Output</div>
            <div class="math-expr">e^zi / Σ e^zj</div>
            <div style="color: #94a3b8; font-size: 12px;">Transforms logits into probability scores across digits 0-9.</div>
        </div>
    """, unsafe_allow_html=True)
