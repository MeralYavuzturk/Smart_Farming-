import tensorflow as tf
from data_loader import veri_hazirla

def model_olustur(num_classes):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),

        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":

    train_dataset, val_dataset = veri_hazirla()

    model = model_olustur(len(train_dataset.class_names))

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=5
    )

    model.save("models/model.h5")
