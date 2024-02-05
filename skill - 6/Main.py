import cv2

import preprocessing as mp
import Model as mm
import numpy as np
import matplotlib.pyplot as plt
import Model as cm

def preprocess_images(images):
    processed_images=[]
    for img in images:
        img = cv2.resize(img,(28,28))
        img = img/255.0
        processed_images.append(img)
    return np.array(processed_images)


def random_mini_batch(X,Y,mini_batch_size=64,seed=None):
    if seed is not None:
        np.random.seed(seed)
    m=X.shape[0] if len(X.shape) > 1 else len(X)
    mini_batches=[];
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,...]
    shuffled_Y = np.array(Y)[permutation]
    num_complete_minibatches=m//mini_batch_size
    for k in range(0,num_complete_minibatches):
        start_idx = k * mini_batch_size
        end_idx=(k+1) * mini_batch_size
        mini_batch_X=shuffled_X[start_idx:end_idx,...]
        mini_batch_y=shuffled_Y[start_idx:end_idx,...]
        mini_batch=(mini_batch_X,mini_batch_y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        start_idx = num_complete_minibatches * mini_batch_size
        end_idx=start_idx+mini_batch_size
        mini_batch_X = shuffled_X[start_idx:end_idx,...]
        mini_batch_y=shuffled_Y[start_idx:end_idx,...]
        mini_batch=(mini_batch_X,mini_batch_y)
        mini_batches.append(mini_batch)
    return mini_batches



# def one_hot_encode(labels):
#     unique_labels=list(set(labels))
#     label_dict={label:i for i , label in enumerate(unique_labels)}
#     numeric_labels=[label_dict[label]for label in labels]
#     num_classes = len(unique_labels)
#     one_hot_encoded = []
#     for numeric_label in numeric_labels:
#         one_hot_vector = [0] * num_classes
#         one_hot_vector[numeric_label] = 1
#         one_hot_encoded.append(one_hot_vector)
#     return one_hot_encoded


if __name__ == "__main__":
    images_folder_path = 'E:\\KLU\\3rd year\\3_2\\deep learning\\Deep Learning Programs\\imagefile'
    imdata = mp.PreProcess_Data()
    imdata.visualization_images(images_folder_path, 2)
    train, label, df = imdata.preprocess(images_folder_path)
    tr_gen, tt_gen,va_gen = imdata.generate_train_test_images(train, label)
    X_train_images = np.array([cv2.imread(img) for img in train])
    x_train_images = preprocess_images(X_train_images)
    Y_train = label

    print(Y_train)

    mini_batches = random_mini_batch(x_train_images, Y_train, mini_batch_size=64,seed=42)

    for mini_batch_X, mini_batch_Y in mini_batches:
        print("x", mini_batch_X.shape)
        print("y", mini_batch_Y.shape)

    image_shape = (28, 28, 3)
    num_classes = 2

    model_adam = cm.DeepANN.simple_model(image_shape, optimizer='adam', num_classes=num_classes)
    model_sgd = cm.DeepANN.simple_model(image_shape, optimizer='sgd', num_classes=num_classes)
    model_rmsprop = cm.DeepANN.simple_model(image_shape, optimizer='rmsprop', num_classes=num_classes)

    cm.compare_model([model_adam, model_sgd, model_rmsprop], tr_gen, va_gen, 5)



























    # CnnModel = mm.DeepANN()
    # model1 = CnnModel.CNN_MODEL_batchNormalization()
    # print("Batch Normalization used in this epochs")
    # print("train generator ", tr_gen)
    # CNN_history = model1.fit(tr_gen, epochs=5, validation_data=va_gen)
    # Cnn_test_loss, Cnn_test_acc = model1.evaluate(tr_gen)
    # # Ann_test_loss, Ann_test_acc = model1.evaluate(tt_gen)
    # print(f'Test accuracy: {Cnn_test_acc}')
    # # model1.save("my_model1.keras")
    # print("The ANN architecture is ")
    # print(model1.summary())

    # CnnModel = mm.DeepANN()
    # model3 = CnnModel.CNN_MODEL_miniBatchNormalization()
    # CNN_history_mini_batch_norm = model3.fit(tr_gen, epochs=5, validation_data=va_gen)

    # Plot CNN_MODEL_miniBatchNormalization training/validation accuracy and loss
    # plt.figure(figsize=(12, 4))
    #
    # plt.subplot(1, 2, 1)
    # # plt.plot(CNN_history_mini_batch_norm.history['val_loss'], label='Validation Loss')
    # plt.plot(CNN_history_mini_batch_norm.history['loss'], label='Training Loss')
    # plt.plot(CNN_history_mini_batch_norm.history['val_accuracy'], label='Validation Accuracy')
    # plt.xlabel('Loss')
    # plt.ylabel('Accuracy')
    # plt.title('CNN_MODEL_miniBatchNormalization Loss and  Accuracy')
    # plt.legend()
    #
    # # plt.subplot(1, 2, 2)
    # # plt.plot(CNN_history_mini_batch_norm.history['loss'], label='Training Loss')
    # # plt.plot(CNN_history_mini_batch_norm.history['val_loss'], label='Validation Loss')
    # # plt.xlabel('Epochs')
    # # plt.ylabel('Loss')
    # # plt.title('CNN_MODEL_miniBatchNormalization Training and Validation Loss')
    # # plt.legend()
    # plt.tight_layout()
    # plt.show()