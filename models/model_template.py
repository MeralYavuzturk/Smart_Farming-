import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = "best_plant_model.keras"
CLASS_INDEX_PATH = "class_indices.json"
IMG_SIZE = (224, 224)

model = load_model(MODEL_PATH)

with open(CLASS_INDEX_PATH, "r", encoding="utf-8") as f:
    class_indices = json.load(f)

idx_to_class = {int(v): k for k, v in class_indices.items()}

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)

    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img, verbose=0)[0]

    class_id = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    print("Tahmin:", idx_to_class[class_id])
    print(f"Güven: %{confidence * 100:.2f}")

    return idx_to_class[class_id], confidence

predict_image("test.jpg")