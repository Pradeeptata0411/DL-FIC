# this is imagumentation for making an image more clearable

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

img_path = r'E:\KLU\3rd year\3_2\deep learning\Deep Learning Programs\Food Classification\momos\001.jpg'
img = image.load_img(img_path, target_size=(150, 150))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir=r'E:\KLU\3rd year\3_2\deep learning\Deep Learning Programs\skill programs\skill-8\agumented images', save_prefix='aug', save_format='jpeg'):
    i += 1
    if i > 30:
        break

original_img = image.load_img(img_path, target_size=(150, 150))
plt.subplot(121)
plt.imshow(original_img)
plt.title('Original Image')

#augmented_img = image.load_img(r'C:\Users\admin\Desktop\images\aug_0_134.jpeg', target_size=(150, 150))
#plt.subplot(122)
#plt.imshow(augmented_img)
#plt.title('Augmented Image')

plt.show()

def display_images_from_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)

        img = imread(image_path)
        plt.imshow(img)
        plt.title(image_file)
        plt.axis('off')
        plt.show()

folder_path=r'E:\KLU\3rd year\3_2\deep learning\Deep Learning Programs\skill programs\skill-8\agumented images'
display_images_from_folder(folder_path)