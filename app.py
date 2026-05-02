import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Smart Farming AI",
    page_icon="🌿",
    layout="wide"
)

# --- 2. CUSTOM CSS (Modern & Neutral Design) ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    [data-testid="stSidebar"] {
        background-color: #1e2124;
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        color: white;
        transform: scale(1.02);
    }
    .result-card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        border-top: 6px solid #2e7d32;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. MODEL & DATA INTEGRATION ---
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("models/best_plant_model.keras")
    with open("models/class_indices (3).json", "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    idx_to_class = {int(v): k for k, v in class_indices.items()}
    return model, idx_to_class

model, idx_to_class = load_assets()

def predict_disease(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(prediction)
    confidence = np.max(prediction)
    full_label = idx_to_class.get(predicted_class_idx, "Unknown___Unknown")
    
    if "___" in full_label:
        plant, disease = full_label.split("___")
    else:
        plant, disease = "Unknown", full_label
    return plant.replace("_", " "), disease.replace("_", " "), float(confidence)

# --- 4. INTERFACE (SIDEBAR) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=80)
    st.title("Project Menu")
    page = st.selectbox("Select Page:", ["🏠 Home", "🔍 Disease Detection"])
    st.markdown("---")
    st.markdown("### 📋 System Status")
    st.success("AI Engine: Online")
    st.success("Database: Connected")

# --- 5. HOME PAGE ---
if page == "🏠 Home":
    st.title("🌿 Smart Farming: Plant Disease Detection")
    st.write("### AI-Powered Agricultural Analysis")
    
    col_text, col_img = st.columns([1, 1])
    with col_text:
        st.write("""
        Our collaborative project focuses on leveraging Deep Learning to monitor crop health. 
        By analyzing leaf patterns, the system provides instant feedback on potential diseases.
        
        **Technical Scope:**
        - CNN-based Image Classification
        - Multi-class Disease Detection
        - Sustainable Agriculture Support
        """)
    with col_img:
        st.image("https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?auto=format&fit=crop&q=80&w=800", use_container_width=True)

# --- 6. DISEASE DETECTION PAGE ---
elif page == "🔍 Disease Detection":
    st.title("🔍 Diagnosis Panel")
    col_u, col_r = st.columns([1, 1], gap="large")

    with col_u:
        st.subheader("📸 Upload Photo")
        uploaded_file = st.file_uploader("Select a leaf image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption='Uploaded Image', use_container_width=True)

    with col_r:
        st.subheader("🧪 Analysis Results")
        if uploaded_file:
            if st.button("Diagnose Disease"):
                with st.spinner('Analyzing patterns...'):
                    plant, disease, score = predict_disease(image)
                
                st.markdown(f"""
                    <div class="result-card">
                        <p style="color:#7f8c8d; font-size:14px; margin-bottom:5px; font-weight:bold;">DETECTION SUMMARY</p>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <h4 style="margin:0; color:#2c3e50; font-size:16px;">Plant Type</h4>
                                <h2 style="margin:0; color:#1b5e20;">{plant}</h2>
                                <br>
                                <h4 style="margin:0; color:#2c3e50; font-size:16px;">Diagnosis</h4>
                                <h2 style="margin:0; color:#e67e22;">{disease}</h2>
                            </div>
                            <div style="text-align:right;">
                                <span style="font-size:32px; font-weight:bold; color:#2ecc71;">{score*100:.1f}%</span>
                                <p style="margin:0; font-size:14px; color:#bdc3c7;">Confidence</p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                st.progress(score)
        else:
            st.info("Please upload a leaf photo to begin analysis.")