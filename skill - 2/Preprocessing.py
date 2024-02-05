import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input
from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np

class PreProcess_Data:
    def visualization_images(self, dir_path, nimages):
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        dpath = dir_path
        count = 0
        for i in os.listdir(dpath):
            train_class = os.listdir(os.path.join(dpath, i))
            for j in range(nimages):
                img_name = train_class[j]
                img_path = os.path.join(dpath, i, img_name)
                img = cv2.imread(img_path)
                axs[count][j].set_title(i)
                axs[count][j].imshow(img)
            count += 1
        fig.tight_layout()
        plt.show(block=True)

    def preprocess(self, dir_path):
        dpath = dir_path
        imagefile = []
        label = []
        for i in os.listdir(dpath):
            train_class = os.listdir(os.path.join(dpath, i))
            for j in train_class:
                img = os.path.join(dpath, i, j)
                imagefile.append(img)
                label.append(i)
        print('Number of train images: {}\n'.format(len(imagefile)))
        print('Number of train image labels: {}\n'.format(len(label)))
        ret_df = pd.DataFrame({'Image': imagefile, 'Labels': label})
        return imagefile, label, ret_df
