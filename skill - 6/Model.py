import tensorflow as tf
from keras.src.layers import Conv2D
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import SGD

class DeepANN():

    def simple_model(self, optimizer, num_classes):
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(units=num_classes, activation='softmax'))

        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        return model



import matplotlib.pyplot as plt

def compare_model(models, train_generator, validate_generator, epochs=5):
    history_list = []
    for model in models:
        print(f"\nTraining model with optimizer: {model.optimizer.get_config()['name']}")
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validate_generator,
            verbose=1
        )
        history_list.append(history)

    # Evaluate on the test set
    test_results = []
    for i, model in enumerate(models):
        print(f"\nEvaluating model with optimizer: {model.optimizer.get_config()['name']}")
        result = model.evaluate(train_generator)
        test_results.append(result)

        print(f"Test Loss for model-{i + 1}: {result[0]}")
        print(f"Test Accuracy for model-{i + 1}: {result[1]}")

    # Plot training accuracy
    for i, history in enumerate(history_list):
        plt.plot(history.history['accuracy'], label=f'Training Accuracy - {models[i].optimizer.get_config()["name"]}')

    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot training loss
    for i, history in enumerate(history_list):
        plt.plot(history.history['loss'], label=f'Training Loss - {models[i].optimizer.get_config()["name"]}')

    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return history_list, test_results
