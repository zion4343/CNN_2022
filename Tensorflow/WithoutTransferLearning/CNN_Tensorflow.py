"""
CNN_Tensorflow
Without TransferLearning
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Activation, BatchNormalization, Flatten, MaxPooling2D
from tensorflow.keras import datasets, optimizers, losses, metrics, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#クラスモデルの実装
class CNN(Model):
  def __init__(self):
    super().__init__()

    self.l1 = Conv2D(filters=32, kernel_size =(3, 3), padding= "same", input_shape = x_train.shape[1:], 
                     kernel_initializer = "he_normal", kernel_regularizer = regularizers.l2(1e-4))
    self.b1 = BatchNormalization()
    self.a1 = Activation("relu")
    self.d1 = Dropout(0.5)

    self.p1 = MaxPooling2D(pool_size = (2, 2))

    self.l2 = Conv2D(filters=64, kernel_size =(3, 3), padding= "same", 
                     kernel_initializer = "he_normal", kernel_regularizer = regularizers.l2(1e-4))
    self.b2 = BatchNormalization()
    self.a2 = Activation("relu")
    self.d2 = Dropout(0.5)

    self.p2 = MaxPooling2D(pool_size = (2, 2))

    self.l3 = Conv2D(filters=128, kernel_size =(3, 3), padding= "same", 
                     kernel_initializer = "he_normal", kernel_regularizer = regularizers.l2(1e-4))
    self.b3 = BatchNormalization()
    self.a3 = Activation("relu")
    self.f3 = Flatten()
    self.d3 = Dropout(0.5)

    self.l4 = Dense(128, kernel_initializer = "he_normal")
    self.b4 = BatchNormalization()
    self.a4 = Activation("relu")

    self.l5 = Dense(10, kernel_initializer = "he_normal", activation = "softmax")

    self.ls = [self.l1, self.b1, self.a1, self.d1, self.p1,
              self.l2, self.b2, self.a2, self.d2, self.p2,
              self.l3, self.b3, self.a3, self.f3, self.d3,
              self.l4, self.b4, self.a4,
              self.l5]
              
  def call(self, x):
    for layer in self.ls:
      x = layer(x)
    return x

#早期終了のクラス
class EarlyStopping:
  def __init__(self, patience = 0, verbose = 0):
    self._stop = 0
    self._loss = float("inf")
    self.patience = patience
    self.verbose = verbose

  def __call__(self, loss):
    if self._loss < loss:
      self._step += 1
      if self._step > self.patience:
        if self.verbose:
          print("early stopping")
        return True

    else:
      self._step = 0
      self.loss = loss

    return False

#1.データの準備
def prepare_data():
  mnist = datasets.cifar10

  (x_train, t_train), (x_test, t_test) = mnist.load_data()

  mean = np.mean(x_train) 
  std = np.std(x_train)

  x_train = (x_train.reshape(-1, 32, 32, 3)/255).astype(np.float32)

  x_test = (x_test.reshape(-1, 32, 32, 3)/255).astype(np.float32)

  x_train, x_test = (x_train - mean)/(std + 1e-7), (x_test - mean)/(std + 1e-7) #標準化

  t_train, t_test = to_categorical(t_train), to_categorical(t_test) #One-Hot 表現に変換

  x_train, x_val, t_train, t_val = train_test_split(x_train, t_train, test_size = 0.2)

  return x_train, x_val, x_test, t_train, t_val, t_test

x_train, x_val, x_test, t_train, t_val, t_test = prepare_data()

#2.モデルの実装
model = CNN()

#3.モデルの学習
criation = losses.CategoricalCrossentropy()
optimizer = optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = True)
train_loss = metrics.Mean()
train_acc = metrics.CategoricalAccuracy()
val_loss = metrics.Mean()
val_acc = metrics.CategoricalAccuracy()

def compute_loss(t, y):
  return criation(t, y)

def train_step(x, t):
  with tf.GradientTape() as tape:
    preds = model(x)
    loss = compute_loss(t, preds)

  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  train_loss(loss)
  train_acc(t, preds)

def val_step(x, t):
  preds = model(x)
  loss = compute_loss(t, preds)
  val_loss(loss)
  val_acc(t, preds)

#学習開始
epochs = 120
batch_size = 64
train_steps = x_train.shape[0]//batch_size
val_steps = x_val.shape[0]//batch_size

hist = {"loss" : [], "accuracy": [], "val_loss": [], "val_accuracy": []}

es = EarlyStopping(patience = 5, verbose = 1)

train_datagen = ImageDataGenerator(featurewise_center = True, featurewise_std_normalization = True,
                                     width_shift_range = 0.1, height_shift_range = 0.1, rotation_range = 10, zoom_range = 0.1,
                                     horizontal_flip=True)

val_datagen = ImageDataGenerator(featurewise_center = True, featurewise_std_normalization = True)

train_datagen.fit(x_train)
val_datagen.fit(x_val)

train_generator = train_datagen.flow(x_train, t_train, batch_size = batch_size)
val_generator = val_datagen.flow(x_val, t_val, batch_size = batch_size)

for epoch in range(epochs):
  t_step_counter = 0
  for x_train_batch, t_train_batch in train_generator:
    train_step(x_train_batch, t_train_batch)
    t_step_counter += 1
    if t_step_counter >= train_steps:
      break

  v_step_counter = 0
  for x_val_batch, t_val_batch in val_generator:
    val_step(x_val_batch, t_val_batch)
    v_step_counter += 1
    if v_step_counter >= val_steps:
      break

  hist["loss"].append(train_loss.result())
  hist["accuracy"].append(train_acc.result())
  hist["val_loss"].append(val_loss.result())
  hist["val_accuracy"].append(val_acc.result())

  print(f"epoch:{epoch+1}, loss:{train_loss.result():.3f}, acc:{train_acc.result():.3f}, val_loss:{val_loss.result():.3f}, val_acc:{val_acc.result():.3f}")

  if es(val_loss.result()):
    break

model.summary()

#4. モデルの評価
#データの可視化
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

#評価
test_loss = metrics.Mean()
test_acc = metrics.CategoricalAccuracy()

def test_step(x, t):
  preds = model(x)
  loss = compute_loss(t, preds)
  test_loss(loss)
  test_acc(t, preds)

test_step(x_test, t_test)