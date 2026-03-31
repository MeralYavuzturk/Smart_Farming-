🌱 Smart Farming : Plant Disease Detection
📌 Project Overview
This project aims to develop an Artificial Intelligence system capable of identifying plant diseases from leaf images. By leveraging deep learning techniques, the system provides automated diagnosis to assist in early intervention and agricultural management.

⚙️ System Architecture
🔹 Input
Image Format: User-provided leaf images (JPEG/PNG).

🔹 Output
disease: The predicted name of the identified plant disease.

confidence: The probability score (confidence level) of the prediction.

📊 Sample Output
JSON
{
  "disease": "Tomato___Early_blight",
  "confidence": 0.99
}
🔗 Backend API & Usage
The frontend or integration layer can utilize the following function to retrieve predictions:

Python
from src.predict import predict_image

# Process the image and get the result
result = predict_image(image)

print(result)
Example Output:
{'disease': 'Tomato___Early_blight', 'confidence': 0.95}

🧠 Technologies Used
Deep Learning: TensorFlow / Keras

Image Processing: OpenCV

Data Analysis: NumPy & Pandas

Visualization: Matplotlib

📁 Project Structure
src/ – Core logic for model architecture and data processing.

dataset/ – Training and validation image datasets.

models/ – Saved pre-trained and final models.

app/ – Interface and deployment files (Future Work).

🚀 Getting Started
1. Environment Setup
Create and activate a virtual environment to manage dependencies:
Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

2. Installation
Install the required libraries:
Bash
pip install -r requirements.txt

3. Model Training
To train the model from scratch, run:
Bash
python src/model.py

4. Running Inference (Prediction)
To make a prediction on a sample image:
Bash
python src/predict.py
