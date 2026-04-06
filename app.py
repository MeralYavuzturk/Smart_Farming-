import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import time

# --- MODEL ENTEGRASYONU ---
@st.cache_resource
def model_yukle():
    return tf.keras.models.load_model("models/model.h5")

model = model_yukle()

# Sınıf isimleri
class_names = [
    'Biber (Bakteriyel Leke)',
    'Biber (Sağlıklı)',
    'Patates (Erken Yanıklık)',
    'Patates (Geç Yanıklık)',
    'Patates (Sağlıklı)',
    'Domates (Bakteriyel Leke)',
    'Domates (Erken Yanıklık)'
]

def tahmin_et(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    return class_names[predicted_class], float(confidence)

# --- ARAYÜZ ---
st.set_page_config(
    page_title="Bitki Sağlık Asistanı",
    page_icon="🌿",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=80)
    st.title("Proje Menüsü")

    sayfa = st.selectbox(
        "Sayfa Seçin:",
        ["🏠 Ana Sayfa", "🔍 Hastalık Tespiti"]
    )

# --- ANA SAYFA ---
if sayfa == "🏠 Ana Sayfa":
    st.title("🌿 Bitki Hastalık Tespit Sistemi")
    st.write("### Hoş Geldiniz")
    st.write("Bu uygulama, tarımda verimliliği artırmak için yapay zeka ile bitki sağlığını kontrol eder.")

    st.image(
        "https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?auto=format&fit=crop&q=80&w=800",
        use_container_width=True
    )

# --- HASTALIK TESPİTİ ---
elif sayfa == "🔍 Hastalık Tespiti":
    st.title("🔍 Hastalık Teşhis Paneli")

    col_u, col_r = st.columns([1, 1])

    with col_u:
        st.subheader("📸 Fotoğraf Yükle")

        yuklenen_dosya = st.file_uploader(
            "Bir yaprak fotoğrafı seçin...",
            type=["jpg", "jpeg", "png"]
        )

        if yuklenen_dosya is not None:
           image = Image.open(yuklenen_dosya)
           image = image.convert("RGB")
           st.image(image, caption='Yüklenen Fotoğraf', use_container_width=True)

    with col_r:
        st.subheader("🧪 Analiz Sonucu")

        if yuklenen_dosya is not None:
            if st.button("Hastalığı Teşhis Et"):
                with st.spinner('Yapay zeka yaprağı inceliyor...'):
                    hastalik, skor = tahmin_et(img)
                    time.sleep(1)

                st.success("✅ Analiz Tamamlandı!")
                st.balloons()

                st.markdown(f"""
<div style="background-color:#e8f5e9; padding:20px; border-radius:10px; border-left: 10px solid #2e7d32;">
<h3 style="color:#1b5e20; margin:0;">Sonuç: {hastalik}</h3>
<p style="color:#2e7d32;">Güven Skoru: %{skor*100:.2f}</p>
</div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Lütfen analiz için bir fotoğraf yükleyin.")