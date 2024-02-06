import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import SGD

class DeepANN():
    def CNN_MODEL(self):
        model = models.Sequential()
        model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(28, 28, 3)))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(2, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer=SGD(), metrics=["accuracy"])
        return model
