from preprocessing import PreProcess_Data
import Model as cm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    images_folder_path = 'E:\\KLU\\3rd year\\3_2\\deep learning\\Deep Learning Programs\\imagefile'
    imdata = PreProcess_Data()
    imdata.visualization_images(images_folder_path, 2)
    imagefile, label, df = imdata.preprocess(images_folder_path)
    tr_gen, tt_gen, va_gen = imdata.generate_train_test_images(imagefile, label)
    num_classes = 2
    input_shape = (28, 28, 3)  # Corrected input shape
    LSTM_MODEL = cm.DeepANN()
    model1 = LSTM_MODEL.Lstm_model(input_shape, num_classes)
    print("train generator ", tr_gen)
    # model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    LSTM_history = model1.fit(tr_gen, epochs=5, validation_data=va_gen)

    plt.plot(LSTM_history.history['accuracy'], label='Training Accuracy')
    plt.plot(LSTM_history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.legend()
    plt.show()

    lstm_test_loss, lstm_test_acc = model1.evaluate(tr_gen)
    print(f'Test accuracy: {lstm_test_acc}')
    print("The ANN architecture is ")
    print(model1.summary())
