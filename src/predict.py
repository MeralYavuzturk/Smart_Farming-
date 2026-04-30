import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# 1. Klasör Yapısına Uygun Dinamik Yollar
# Bu satır, src içindeki kodun bir üst klasöre çıkıp models'e bakmasını sağlar
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_plant_model.keras")
JSON_PATH = os.path.join(BASE_DIR, "models", "class_indices (3).json")

# 2. Modeli ve Sınıf Listesini Yükle
# Hata almamak için Merve'nin .keras dosyası ve JSON listesini yüklüyoruz
model = tf.keras.models.load_model(MODEL_PATH)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    class_indices = json.load(f)

# JSON'daki "0": "Hastalık" yapısını modele uygun hale getiriyoruz
#idx_to_class = {int(k): v for k, v in class_indices.items()}
idx_to_class = {int(v): k for k, v in class_indices.items()}

def predict_image(image):
    """
    Zelal'in (Frontend) göndereceği PIL Image nesnesini alıp tahmin döndürür.
    """
    # Resmi modelin beklediği boyuta getiriyoruz
    img = image.resize((224, 224))
    
    # Bazı resimler 4 kanal (RGBA) olabilir, 3 kanala (RGB) çeviriyoruz
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    # Normalizasyon: 0-255 arası değerleri 0-1 arasına çekiyoruz
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Tahmini gerçekleştir
    prediction = model.predict(img, verbose=0)
    
    predicted_class_idx = np.argmax(prediction)
    confidence = np.max(prediction)

    return {
        "disease": idx_to_class.get(predicted_class_idx, "Bilinmeyen"),
        "confidence": float(confidence)
    }

if __name__ == "__main__":
    # Test klasöründeki testimage.jpg ile deneme yapıyoruz
    test_img_path = os.path.join(BASE_DIR, "testimage.jpg")
    
    if os.path.exists(test_img_path):
        img = Image.open(test_img_path)
        result = predict_image(img)
        print(f"\n--- TEST SONUCU ---")
        print(f"Hastalık: {result['disease']}")
        print(f"Güven Oranı: %{result['confidence']*100:.2f}")
    else:
        print(f"\nHata: {test_img_path} bulunamadı!")