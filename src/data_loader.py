import tensorflow as tf

def veri_hazirla():

    print("Train ve Validation verileri yükleniyor...")
    
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "dataset/train",
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(224, 224),
        batch_size=32
    )

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "dataset/train",
        validation_split=0.2,
        subset="validation", 
        seed=123,
        image_size=(224, 224),
        batch_size=32
    )
    print("Sınıflar:", train_dataset.class_names)

    return train_dataset, val_dataset
   

if __name__ == "__main__":
  veri_hazirla()