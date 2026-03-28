import tensorflow as tf

def veri_hazirla(data_yolu):
    # Bu fonksiyon Merve ve Ayşe veri setini yüklediğinde 
    # resimleri modele pompalayacak.
    print(f"--- {data_yolu} klasörü taranıyor...")
    
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_yolu,
        image_size=(224, 224),
        batch_size=32
    )

    print("Veri başarıyla yüklendi!")
    print("Sınıflar:", dataset.class_names)

    return dataset

if __name__ == "__main__":
    veri_hazirla("dataset")