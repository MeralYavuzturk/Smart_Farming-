import tensorflow as tf
import os
import matplotlib.pyplot as plt

def get_callbacks(model_name="bitki_hastalik_modeli.h5"):
    """
    Eğitimi denetleyen yardımcıları (Callbacks) döndürür.
    """
    # Kayıt yolu: models/ klasörü
    checkpoint_path = os.path.join("models", model_name)
    
    # 1. ModelCheckpoint: En iyi başarıyı gördüğünde modeli otomatik kaydeder
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    # 2. CSVLogger: Her turun (epoch) sonucunu data/ klasörüne rapor olarak yazar
    log_path = os.path.join("data", "egitim_raporu.csv")
    csv_logger = tf.keras.callbacks.CSVLogger(log_path, append=True)
    
    return [checkpoint, csv_logger]

def plot_training_results(history):
    """
    Eğitim bittiğinde Accuracy ve Loss grafiklerini çizer.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    
    # Başarı Grafiği
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Eğitim Başarısı')
    plt.plot(epochs_range, val_acc, label='Doğrulama Başarısı')
    plt.title('Model Başarı Oranı')
    plt.legend(loc='lower right')

    # Kayıp Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Eğitim Kaybı')
    plt.plot(epochs_range, val_loss, label='Doğrulama Kaybı')
    plt.title('Model Kayıp Oranı')
    plt.legend(loc='upper right')
    
    plt.show()