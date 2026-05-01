import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import time 

# Base paths
# This line allows the code inside src to go up one folder and look at models


# 1. Klasör Yapısına Uygun Dinamik Yollar
# Bu satır, src içindeki kodun bir üst klasöre çıkıp models'e bakmasını sağlar

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_plant_model.keras")
JSON_PATH = os.path.join(BASE_DIR, "models", "class_indices (3).json")



def load_model_and_classes():
    """
    Loads the trained model and class mappings.
    """
    try:
       model = tf.keras.models.load_model(MODEL_PATH)

       with open(JSON_PATH, "r", encoding="utf-8") as f:
           class_indices = json.load(f)

       idx_to_class = {int(v): k for k, v in class_indices.items()}

       return model, idx_to_class
    
    except Exception as e:
        print("Model yüklenirken hata oluştu:",e)
        return None,None

# Load model globally (only once)     
model, idx_to_class = load_model_and_classes()

if model is None:
    raise Exception("Model yüklenemedi, sistem durduruldu.")

# Warm-up (dummy input)
dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
_ = model(dummy, training=False)



def predict_image(image):
    """
    Takes a PIL Image and returns prediction result.
    """
    # bringing it to the size expected by the official model
    try:
        img = image.resize((224, 224))
    
        # Some images may be 4 channel (RGBA), we convert them to 3 channels (RGB)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Normalization: We bring values between 0-255 to the range of 0-1
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

    # model timing 
        start_time = time.time()

        # Perform the estimate
        prediction = model(img, training=False).numpy()
        end_time = time.time() #finish the time

        predicted_class_idx = np.argmax(prediction)
        confidence = np.max(prediction)
    
        inference_time = end_time - start_time

        return {
            "disease": idx_to_class.get(predicted_class_idx, "Bilinmeyen"),
            "confidence": float(confidence),
            "response_time": inference_time
        }
    except Exception as e:
        return {
            "error": str(e)
        }

if __name__ == "__main__":
    # testing with testimage.jpg in the test folder
    test_img_path = os.path.join(BASE_DIR, "testimage.jpg")
    
    if os.path.exists(test_img_path):
        try:
           img = Image.open(test_img_path)
           result = predict_image(img)

           print(f"\n--- TEST SONUCU ---")
           # If predict returned an error
           if "error" in result:
               print("Hata oluştu:", result["error"])
           else:
            print(f"Hastalık: {result['disease']}")
            print(f"Güven Oranı: %{result['confidence']*100:.2f}")
            print(f"Response Time: {result['response_time']:.4f} seconds")

            # Average inference time test
            img_resized = img.resize((224, 224))
            if img_resized.mode != 'RGB':
                img_resized = img_resized.convert('RGB')

            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

           times = []
           for _ in range(5):
               start = time.time()
               _ = model(img_array, training=False)
               end = time.time()
               times.append(end - start)

           avg_time = sum(times) / len(times)
           print(f"Average Response Time: {avg_time:.4f} seconds")
        
        except Exception as e:
            print("Test sirasinda hata oluştu:",e)
    else:
        print(f"\nHata: {test_img_path} bulunamadi!")

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

