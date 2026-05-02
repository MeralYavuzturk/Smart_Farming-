import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Smart Farming AI",
    page_icon="🌿",
    layout="wide"
)

# --- MODEL & DATA INTEGRATION ---
# Update: Loading the new .keras model and JSON categories
@st.cache_resource
def load_assets():
    # Make sure these files are in your 'models' folder
    model = tf.keras.models.load_model("models/best_plant_model.keras")
    
    # We load the JSON to get the correct English labels from your team
    with open("models/class_indices (3).json", "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    
    # Mapping the indices correctly
    idx_to_class = {int(v): k for k, v in class_indices.items()}
    return model, idx_to_class

model, idx_to_class = load_assets()

def predict_disease(image):
    # Preprocessing for the new model
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(prediction)
    confidence = np.max(prediction)

    full_label = idx_to_class.get(predicted_class_idx, "Unknown___Unknown")
    
    # Splitting the label for the "Plant" and "Disease" cards (Sprint 3 requirement)
    if "___" in full_label:
        plant, disease = full_label.split("___")
    else:
        plant, disease = "Unknown", full_label
        
    return plant.replace("_", " "), disease.replace("_", " "), float(confidence)

# --- INTERFACE (SIDEBAR) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=80)
    st.title("Project Menu")
    
    # Fully English Menu
    page = st.selectbox(
        "Select Page:",
        ["🏠 Home", "🔍 Disease Detection"]
    )

# --- HOME PAGE ---
if page == "🏠 Home":
    st.title("🌿 Smart Farming: Plant Disease Detection")
    st.write("### Welcome to the AI Assistant")
    st.write("""
    This application utilizes Artificial Intelligence to monitor plant health and increase agricultural efficiency. 
    Our system can identify various diseases from a single leaf photo.
    """)

    st.image(
        "https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?auto=format&fit=crop&q=80&w=800",
        use_container_width=True
    )

# --- DISEASE DETECTION PAGE ---
elif page == "🔍 Disease Detection":
    st.title("🔍 Diagnosis Panel")

    col_u, col_r = st.columns([1, 1])

    with col_u:
        st.subheader("📸 Upload Photo")
        uploaded_file = st.file_uploader(
            "Select a leaf image...",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
           image = Image.open(uploaded_file)
           image = image.convert("RGB")
           st.image(image, caption='Target Leaf Image', use_container_width=True)

    with col_r:
        st.subheader("🧪 Analysis Results")

        if uploaded_file is not None:
            if st.button("Diagnose Disease"):
                with st.spinner('AI is analyzing the leaf patterns...'):
                    plant, disease, score = predict_disease(image)

                st.success("✅ Analysis Complete!")
                
                # Sprint 3: Detailed Result Cards
                st.markdown(f"""
                <div style="background-color:#f1f8e9; padding:20px; border-radius:10px; border-left: 10px solid #2e7d32; margin-bottom:10px;">
                    <p style="color:#555; margin:0;">PLANT TYPE</p>
                    <h2 style="color:#1b5e20; margin:0;">{plant}</h2>
                </div>
                <div style="background-color:#fff3e0; padding:20px; border-radius:10px; border-left: 10px solid #ef6c00;">
                    <p style="color:#555; margin:0;">DIAGNOSIS (DISEASE)</p>
                    <h2 style="color:#e65100; margin:0;">{disease}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.write(f"**Confidence Score:** {score*100:.2f}%")
                st.progress(score)
        else:
            st.warning("Please upload a photo to start the diagnosis.")