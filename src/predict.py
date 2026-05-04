import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import time

# 1. Dynamic Path Management
# Ensures the code can find models folder regardless of where it's run
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_plant_model.keras")
JSON_PATH = os.path.join(BASE_DIR, "models", "class_indices (3).json")

def load_model_and_classes():
    """
    Loads the trained model (.keras) and class mappings (JSON).
    """
    try:
        # Loading the new .keras model format as requested by the team
        model = tf.keras.models.load_model(MODEL_PATH)

        with open(JSON_PATH, "r", encoding="utf-8") as f:
            class_indices = json.load(f)

        # Map index to class name
        idx_to_class = {int(v): k for k, v in class_indices.items()}
        return model, idx_to_class
    
    except Exception as e:
        print(f"Error loading model or JSON: {e}")
        return None, None

# Load model globally (optimization: loads only once)
model, idx_to_class = load_model_and_classes()

if model is None:
    raise Exception("Model could not be loaded. System halted.")

# Warm-up (dummy input to initialize the engine)
dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
_ = model(dummy, training=False)

def predict_image(image):
    """
    Processes a PIL Image and returns prediction results in English.
    """
    try:
        # Preprocessing: Resize to 224x224 as per Sprint 1 requirements
        img = image.resize((224, 224))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Normalization: 0-1 range
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Inference and Timing
        start_time = time.time()
        prediction = model(img_array, training=False).numpy()
        end_time = time.time()

        predicted_class_idx = np.argmax(prediction)
        confidence = np.max(prediction)
        inference_time = end_time - start_time

        # Result mapping
        full_label = idx_to_class.get(predicted_class_idx, "Unknown")
        
        # Split "Plant___Disease" for better UI display (Sprint 3 task)
        if "___" in full_label:
            plant_name, disease_name = full_label.split("___")
        else:
            plant_name, disease_name = "Unknown", full_label

        return {
            "plant": plant_name.replace("_", " "),
            "disease": disease_name.replace("_", " "),
            "confidence": float(confidence),
            "response_time": inference_time
        }
    except Exception as e:
        return {"error": str(e)}

# Test block for local debugging
if __name__ == "__main__":
    test_img_path = os.path.join(BASE_DIR, "testimage.jpg")
    
    if os.path.exists(test_img_path):
        img = Image.open(test_img_path)
        result = predict_image(img)
        print("\n--- TEST RESULTS ---")
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Plant: {result['plant']}")
            print(f"Status/Disease: {result['disease']}")
            print(f"Confidence: {result['confidence']*100:.2f}%")
            print(f"Response Time: {result['response_time']:.4f}s")
    else:

        print(f"Test image not found at: {test_img_path}")

