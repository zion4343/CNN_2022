"""
CNN - Tensorflow with Transfer Learning
犬の犬種を判別できるようモデルを学習させる
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Activation, BatchNormalization, Flatten, MaxPooling2D, GlobalAveragePooling2D, Concatenate
from tensorflow.keras import datasets, optimizers, losses, metrics, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os, glob, random, math
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler


from tensorflow.keras.applications import vgg16, xception, inception_v3, inception_resnet_v2, resnet_v2, nasnet
from tensorflow.keras.layers.experimental import preprocessing

#読み込む画像サイズ
IMG_HEIGHT = 331
IMG_WIDTH = 331

#クラスモデルの実装
class CNN(Model):
  def __init__(self):
    super().__init__()

    #layer1: Xception
    transfer_1 = xception.Xception(weights = "imagenet", include_top = False, input_shape = (IMG_HEIGHT, IMG_WIDTH, 3))
    transfer_1.trainable = False

    #layer2: InceptionV3
    transfer_2 = inception_v3.InceptionV3(weights = "imagenet", include_top = False, input_shape = (IMG_HEIGHT, IMG_WIDTH, 3))
    transfer_2.trainable = False

    #layer3: InceptionResNetV2
    transfer_3 = inception_resnet_v2.InceptionResNetV2(weights = "imagenet", include_top = False, input_shape = (IMG_HEIGHT, IMG_WIDTH, 3))
    transfer_3.trainable = False

    #layer4: NASNetLarge
    transfer_4 = nasnet.NASNetLarge(weights = "imagenet", include_top = False, input_shape = (IMG_HEIGHT, IMG_WIDTH, 3))
    transfer_4.trainable = False

    self.l1 = transfer_1
    self.p1 = GlobalAveragePooling2D()
    self.l2 = transfer_2
    self.p2 = GlobalAveragePooling2D()
    self.l3 = transfer_3
    self.p3 = GlobalAveragePooling2D()
    self.l4 = transfer_4
    self.p4 = GlobalAveragePooling2D()

    self.l5 = Dense(512, kernel_initializer = "he_normal")
    self.b5 = BatchNormalization()
    self.a5 = Activation("relu")

    self.l6 = Dense(120, activation = "softmax")

    self.ls = [self.l5, self.b5, self.a5,
               self.l6]

  def call(self, x):
    x1 = xception.preprocess_input(x)
    x1 = self.l1(x1)
    x1 = self.p1(x1)
    x2 = inception_v3.preprocess_input(x)
    x2 = self.l2(x2)
    x2 = self.p2(x2)
    x3 = inception_resnet_v2.preprocess_input(x)
    x3 = self.l3(x3)
    x3 = self.p3(x3)
    x4 = nasnet.preprocess_input(x)
    x4 = self.l1(x4)
    x4 = self.p1(x4)

    x = Concatenate()([x1, x2, x3, x4])

    for layer in self.ls:
      x = layer(x)
    return x

#1. データの準備
def prepare_data():
  #データの読み込み
  train_data = np.load("drive/MyDrive/Colab Notebooks/Images/train_dogs_data.npz")
  x_train = train_data["x"]
  t_train = train_data["y"]

  val_data = np.load("drive/MyDrive/Colab Notebooks/Images/val_dogs_data.npz")
  x_val = val_data["x"]
  t_val = val_data["y"]

  #データの加工処理
  batch_size = 128

  train_datagen = ImageDataGenerator(featurewise_center = True, featurewise_std_normalization = True,
                                     width_shift_range = 0.1, height_shift_range = 0.1, rotation_range = 10, zoom_range = 0.1,
                                     horizontal_flip=True)

  val_datagen = ImageDataGenerator(featurewise_center = True, featurewise_std_normalization = True)

  train_datagen.fit(x_train)
  val_datagen.fit(x_val)

  train_generator = train_datagen.flow(x_train, t_train, batch_size = batch_size)
  val_generator = val_datagen.flow(x_val, t_val, batch_size = batch_size)

  return train_generator, val_generator

#2.モデルの実装

#3.モデルの学習
def train(train_generator, val_generator):
  model = CNN()
  optimizer = optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = True)

#学習開始
  epochs = 50
  batch_size = len(train_generator[0][0])

  total_train = len(train_generator) * batch_size
  steps_per_epoch = total_train // batch_size
  total_validate = len(val_generator) * batch_size
  validation_steps = total_validate // batch_size

  es = EarlyStopping(monitor = "val_loss", patience = 10, verbose = 1)

  model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])

  hist = model.fit(train_generator, epochs = epochs, verbose = 1, validation_data = val_generator,
                   steps_per_epoch = steps_per_epoch, validation_steps = validation_steps,
                   callbacks = [es])

  return model, hist

#4. モデルの評価
#データの可視化
def plot_hist(hist):
  loss = hist["loss"]
  val_loss = hist["val_loss"]
  fig = plt.figure()
  plt.rc("font", family = "serif")
  plt.plot(range(len(loss)), loss, color = "black", linewidth = 1, label ="loss(Training)")
  plt.plot(range(len(val_loss)), val_loss, color = "red", linewidth = 1, label ="loss(Validation)")
  plt.legend(loc="best")
  plt.grid()
  plt.xlabel("epochs")
  plt.ylabel("loss")
  plt.show()

  acc = hist["accuracy"]
  val_acc = hist["val_accuracy"]
  fig = plt.figure()
  plt.rc("font", family = "serif")
  plt.plot(range(len(acc)), acc, color = "black", linewidth = 1, label ="accuracy(Training)")
  plt.plot(range(len(val_acc)), val_acc, color = "red", linewidth = 1, label ="accuracy(Validation)")
  plt.legend(loc="best")
  plt.grid()
  plt.xlabel("epochs")
  plt.ylabel("accuracy")
  plt.show()

#実際に動かす
train_generator, val_generator = prepare_data()

model, hist = train(train_generator, val_generator)

plot_hist(hist)

#学習結果を保存
with open("dog_breed_model.json", "w") as json_file:
  json_file.write(model.to_json())

model.save_weights("dog_breed_weight.h5")

print("")