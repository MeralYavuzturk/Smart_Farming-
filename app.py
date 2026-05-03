import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os
import time
from tensorflow.keras.applications.efficientnet import preprocess_input

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

    /* Buton Tasarımı */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;

         background-color: #2e7d32;
        color: white;
        font-weight: bold;
        border: none;

    }
    /* Sonuç Kartı Tasarımı (Görsel 3'teki beyaz kart) */
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        color: #2c3e50;
        margin-top: 15px;
        border-left: 10px solid #2e7d32;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        background-color: #1b2e1c;
        color: #4caf50;
        border: 1px solid #2e7d32;
        font-size: 14px;

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

# --- 🔥 TECHNICAL BACKEND (Meral & Peri) ---
def predict_disease(image):
    try:
        start_time = time.time()
        
        # Ön işleme adımları
        img = image.resize((224, 224))
        img = img.convert("RGB")
        img_array = np.array(img, dtype=np.float32)
        
        # EfficientNet için kritik ön işleme
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Tahmin
        prediction = model.predict(img_array, verbose=0)
        end_time = time.time()
        
        idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        full_label = idx_to_class.get(idx, "Unknown___Unknown")
        
        if "___" in full_label:
            plant, disease = full_label.split("___")
        else:
            plant, disease = "Unknown", full_label
            
        return plant.replace("_", " "), disease.replace("_", " "), confidence, (end_time - start_time)
    except Exception as e:
        return "Error", str(e), 0.0, 0.0

# --- 4. SIDEBAR (Görsel 1 & 2'deki tasarım) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=100)
    st.title("Project Menu")
    st.write("Select Page:")
    page = st.selectbox("", ["🏠 Home", "🔍 Disease Detection"], label_visibility="collapsed")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### 📋 System Status")
    st.markdown('<div class="status-box">AI Engine: Online</div>', unsafe_allow_html=True)
    st.markdown('<div class="status-box">Database: Connected</div>', unsafe_allow_html=True)

# --- 5. HOME PAGE (Görsel 1) ---

if page == "🏠 Home":
    st.title("🌿 Smart Farming: Plant Disease Detection")
    st.write("### AI-Powered Agricultural Analysis")
    

   col_text, col_img = st.columns([1, 1.2])
    with col_text:
       st.markdown("""
        *   **📸 Upload Photo:** Submit an image of a plant leaf to the system.
        *   **🔍 Instant Diagnosis:** See which disease the leaf has.
        *   **✅ Apply With Confidence:** Be assured of accurate diagnosis with a high rate of success.
        """)
    with col_img:
        # Görseldeki bitki fotoğrafı
        st.image("https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?auto=format&fit=crop&q=80&w=800", use_container_width=True)

# --- 6. DISEASE DETECTION PAGE (Görsel 2 & 3) ---
elif page == "🔍 Disease Detection":
    st.title("🔍 Diagnosis Panel")
    
    col_u, col_r = st.columns([1, 1], gap="large")

    with col_u:
        st.markdown("### 📸 Upload Photo")
        st.write("Select a leaf image...")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=uploaded_file.name, use_container_width=True)

    with col_r:
        st.markdown("### 🧪 Analysis Results")
        if uploaded_file:
            if st.button("Diagnose Disease"):
                with st.spinner('Analyzing patterns...'):
                    plant, disease, score, res_time = predict_disease(image)
                
                # Görsel 3'teki beyaz sonuç kartı tasarımı
                st.markdown(f"""
                    <div class="result-card">
                        <p style="color:#7f8c8d; font-size:12px; font-weight:bold; margin-bottom:5px;">DETECTION SUMMARY</p>
                        <div style="display: flex; justify-content: space-between;">
                            <div>
                                <p style="margin:0; font-size:14px;">Plant Type</p>
                                <h2 style="margin:0; color:#1b5e20; font-size:28px;">{plant}</h2>
                                <br>
                                <p style="margin:0; font-size:14px;">Diagnosis</p>
                                <h2 style="margin:0; color:#e67e22; font-size:28px;">{disease}</h2>
                            </div>
                            <div style="text-align:right;">
                                <h1 style="margin:0; color:#2ecc71; font-size:40px;">{score*100:.1f}%</h1>
                                <p style="margin:0; color:#bdc3c7; font-size:12px;">Confidence</p>
                                <p style="margin-top:10px; color:#bdc3c7; font-size:11px;">Hız: {res_time:.4f}s</p>

                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                st.progress(score)
        else:

 
            st.info("Please upload a leaf photo to begin analysis.")