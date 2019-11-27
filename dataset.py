import os
import random
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input


my_path = list(paths.list_images('train/'))
# print(my_path)reset

random.shuffle(my_path)

labels = [p.split(os.path.sep)[-2] for p in my_path]
# print(labels, len(labels))

le = LabelEncoder()
labels = le.fit_transform(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

list_images = []
for (i, image_path) in enumerate(my_path):
  img = load_img(image_path, target_size=(224, 224))
  img = img_to_array(img)

  img = np.expand_dims(img, 0)
  img = imagenet_utils.preprocess_input(img)

  list_images.append(img)

list_images = np.vstack(list_images)

X_train, X_test, y_train, y_test = train_test_split(list_images, labels, test_size=0.2, random_state=42)

train_datagen = ImageDataGenerator(
  rescale=1. / 255,
  rotation_range=30,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.2,
  zoom_range=0.2, horizontal_flip=True,
  fill_mode='nearest'
)

# train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_datagen = ImageDataGenerator(rescale=1. /255)

train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
# train_generator = train_datagen.flow_from_directory(
#   './train',
#   target_size=(224, 224),
#   color_mode='rgb',
#   batch_size=32,
#   class_mode='categorical',
#   shuffle=True
# )

test_generator = test_datagen.flow(X_test, y_test, batch_size=32)
