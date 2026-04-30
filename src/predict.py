import tensorflow as tf
import numpy as np
from PIL import Image

# modeli yükle
model = tf.keras.models.load_model("models/model.h5")

# class isimleri (data_loader ile aynı sırada olmalı)
class_names = [
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight'
]

def predict_image(image):

    
    img = image.resize((224, 224))

    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    return {
        "disease": class_names[predicted_class],
        "confidence": float(confidence)
    }


if __name__ == "__main__":
    img = Image.open("testimage.jpg")  # sadece test için
    result = predict_image(img)
    print(result)