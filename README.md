# 🌱 PLANT DISEASE DETECTION SYSTEM (SMART FARMING)

## This project is an end-to-end software solution that aims to detecet diseases in plant leaves using image analysis methods with deep learning techniques.

## 🎯 Project Purpose

By analyzing plant leaf photos uploaded by users;

To determine whether the plant is healthy or diseased,

İf it is diseased,to diagnose the type of disease,

To establish a rapid decision support mechanism for farmers and hobby gardens.

🏗️ Project Architecture & Folder Structure
The project is built on a modular and scalable backend architecture:
├── app/              # Frontend: Streamlit-based user interface
├── data/             # Raporting: Training logs and performance data (CSV)
├── dataset/          # Data Set: Train, Validation and Test folders
├── models/           # Storage: Best saved model files (.h5)
├── notebooks/        # Ar-Ge: Model testing and data analysis
└── src/              # Backend: The main processing engine of the project
    ├── data_loader.py    # Data loading and preprocessing operations
    ├── train_helper.py   # Callbacks, Raporting and Model Loading 
    └── sistem_kontrol.py # Environment and library validation

🚀 Installation and Operation
To run the project in your own environment:

Clone the Repository: git clone [https://github.com/MeralYavuzturk/Smart_Farming-]

Create the virtual environment: python -m venv env

Install the libraries: pip install -r requirements.txt

Start the interface: streamlit run app/main.py

👥 Project Team
Meral YAVUZTÜRK
Zelal ERGİN
Ayşe MUTLUAY
Merve ÖZCAN 
Perihan ÇELİKOĞLU

