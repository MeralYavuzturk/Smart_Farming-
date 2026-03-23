import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

print("\n" + "="*30)
print("🚀 PROJE SİSTEM KONTROLÜ")
print("="*30)

try:
    # 1. TensorFlow Kontrolü (Yapay Zeka Beyni)
    print(f"✅ TensorFlow Hazır: {tf.__version__}")
    
    # 2. OpenCV Kontrolü (Görüntü İşleme Gözü)
    print(f"✅ OpenCV Hazır: {cv2.__version__}")
    
    # 3. Pandas & NumPy (Veri ve Matematik)
    test_data = pd.DataFrame({'Deneme': [1, 2, 3]})
    print(f"✅ Pandas & NumPy Hazır: {pd.__version__}")

    # 4. Streamlit (Arayüz vitrini)
    print(f"✅ Streamlit Hazır: {st.__version__}")

    # 5. Görselleştirme Testi (Küçük Bir Grafik Çizelim)
    print("📊 Grafik motoru test ediliyor...")
    plt.plot([1, 2, 3], [1, 4, 9])
    plt.title("Sistem Çalışıyor!")
    plt.show()

    print("\n🎉 TEBRİKLER! Tüm kütüphaneler hatasız kurulmuş.")
    print("="*30)

except Exception as e:
    print(f"\n❌ HATA OLUŞTU: {e}")
    print("Lütfen eksik kütüphaneyi tekrar yükleyin.")