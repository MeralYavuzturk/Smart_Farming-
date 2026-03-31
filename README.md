🌱 Akıllı Tarım - Bitki Hastalık Tespiti
📌 Proje Amacı

Bu proje, bitki yapraklarından hastalık tespiti yapan bir yapay zeka sistemi geliştirmeyi amaçlamaktadır.

⚙️ Sistem Mimarisi
🔹 Input (Girdi)
Kullanıcıdan alınan yaprak görüntüsü (jpg/png)
🔹 Output (Çıktı)
disease: Tahmin edilen hastalık adı
confidence: Modelin tahmin güven oranı

📊 Örnek Çıktı
{
  "disease": "Tomato___Early_blight",
  "confidence": 0.99
}

🧠 Kullanılan Teknolojiler
TensorFlow / Keras
OpenCV
NumPy & Pandas
Matplotlib

📁 Proje Yapısı
src/ → model ve veri işlemleri
dataset/ → eğitim verisi
models/ → eğitilmiş model
app/ → arayüz (ileride)

🚀 Çalıştırma
1. Ortam oluştur
python -m venv venv
venv\Scripts\activate
2. Kütüphaneleri yükle
pip install -r requirements.txt
3. Model eğit
python src/model.py
4. Tahmin yap
python src/predict.py
