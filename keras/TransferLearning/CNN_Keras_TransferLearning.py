"""
CNN - Keras
With Transfer Learning
"""

#Keras

import math
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Activation, BatchNormalization, Flatten, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras import datasets, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import VGG16


'''
1. データの準備
'''
def prepare_data():
  #データのダウンロード
  _URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
  path_to_zip = tf.keras.utils.get_file("cats_and_dogs.zip", origin = _URL, extract = True)
  PATH = os.path.join(os.path.dirname(path_to_zip), "cats_and_dogs_filtered")

  train_dir = os.path.join(PATH, "train")
  val_dir = os.path.join(PATH, "validation")

  train_cats_dir = os.path.join(train_dir, "cats")
  train_dogs_dir = os.path.join(train_dir, "dogs")
  val_cats_dir = os.path.join(val_dir, "cats")
  val_dogs_dir = os.path.join(val_dir, "dogs")

  #データの加工処理
  batch_size = 32
  IMG_HEIGHT = 224
  IMG_WIDTH = 224

  train_image_generator = ImageDataGenerator(rescale = 1./255, rotation_range = 15, shear_range = 0.2, 
                                                zoom_range = 0.2, horizontal_flip = True, fill_mode = "nearest",
                                                width_shift_range = 0.1, height_shift_range = 0.1)

  val_image_generator = ImageDataGenerator(rescale = 1./255)

  train_data_gen = train_image_generator.flow_from_directory(batch_size = batch_size, directory = train_dir,
                                                                shuffle = True, target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                class_mode = "binary")

  val_data_gen = val_image_generator.flow_from_directory(batch_size = batch_size, directory = val_dir,
                                                            shuffle = True, target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                            class_mode = "binary")

  return train_data_gen, val_data_gen


'''
2. モデルの実装
'''
def model_FClayer():
  image_size = len(train_data_gen[0][0][0])
  input_shape = (image_size, image_size, 3)

  pre_trained_model = VGG16(include_top = False, weights = "imagenet", input_shape = input_shape)

  for layer in pre_trained_model.layers[:15]:
    layer.trainable = False

  for layer in pre_trained_model.layers[15:]:
    layer.trainable = True


  model = Sequential()

  model.add(pre_trained_model)

  model.add(GlobalMaxPooling2D())

  model.add(Dense(512, kernel_initializer = "he_normal"))
  model.add(BatchNormalization())
  model.add(Activation("relu"))

  model.add(Dense(1, activation = "sigmoid"))

  return model


'''
3. モデルの学習
'''
def train_FClayer(train_data_gen, val_data_gen):
  model = model_FClayer()
  optimizer = optimizers.RMSprop(learning_rate = 1e-5, rho = 0.99)
  epochs = 40
  batch_size = len(train_data_gen[0][0])

  total_train = len(train_data_gen) * batch_size
  steps_per_epoch = total_train // batch_size
  total_validate = len(val_data_gen) * batch_size
  validation_steps = total_validate // batch_size

  def step_decay(epoch):
    initial_lrate = 1e-5
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((epoch)/epochs_drop))
    return lrate

  lrate = LearningRateScheduler(step_decay)

  model.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])

  hist = model.fit(train_data_gen, epochs = epochs, verbose = 1, validation_data = val_data_gen,
                   steps_per_epoch = steps_per_epoch, validation_steps = validation_steps,
                   callbacks = [lrate])
  
  return model, hist


'''
4. モデルの評価
'''
def plot_hist(hist):
  loss = hist.history["loss"]
  val_loss = hist.history["val_loss"]
  fig = plt.figure()
  plt.rc("font", family = "serif")
  plt.plot(range(len(loss)), loss, color = "black", linewidth = 1, label ="loss(Training)")
  plt.plot(range(len(val_loss)), val_loss, color = "red", linewidth = 1, label ="loss(Validation)")
  plt.legend(loc="best")
  plt.grid()
  plt.xlabel("epochs")
  plt.ylabel("loss")
  plt.show()

  acc = hist.history["accuracy"]
  val_acc = hist.history["val_accuracy"]
  fig = plt.figure()
  plt.rc("font", family = "serif")
  plt.plot(range(len(acc)), acc, color = "black", linewidth = 1, label ="accuracy(Training)")
  plt.plot(range(len(val_acc)), val_acc, color = "red", linewidth = 1, label ="accuracy(Validation)")
  plt.legend(loc="best")
  plt.grid()
  plt.xlabel("epochs")
  plt.ylabel("accuracy")
  plt.show()

  lr = hist.history["lr"]
  fig = plt.figure()
  plt.rc("font", family = "serif")
  plt.plot(range(len(lr)), lr, color = "black", linewidth = 1, label ="Learning Rate")
  plt.legend(loc="best")
  plt.grid()
  plt.xlabel("epochs")
  plt.ylabel("learning rate")
  plt.show()

#実際に学習させる
train_data_gen, val_data_gen = prepare_data()
model, hist = train_FClayer(train_data_gen, val_data_gen)
plot_hist(hist)

#学習結果を保存
with open("cats_or_dog_model.json", "w") as json_file:
  json_file.write(model.to_json())

model.save_weights("cats_or_dog_weight.h5")

print("")