'''
CNN
Load and Identify cats or dog
'''

from os import pread
from IPython.core.display import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from PIL import Image
import matplotlib.pyplot as plt

#モデルの読み込み
model = model_from_json(open("cats_or_dog_model.json", "r").read())

model.load_weights("cats_or_dog_weight.h5")

#描画する関数
def draw(x):
  plt.figure(figsize = (10, 10))
  pos = 1
  for i in range(x.shape[0]):
    plt.subplot(4, 5, pos)
    plt.imshow(x[i])
    plt.axis("off")
    pos += 1
  plt.show()

#データの準備
IMG_HEIGHT = 224
IMG_WIDTH = 224

img = Image.open("Haro_dog.jpeg") #カラーで読み込み
img = img.convert("RGB")
img = img.resize((IMG_HEIGHT, IMG_WIDTH))

x = np.asarray(img) #データに変換
x = x.reshape(-1, IMG_WIDTH, IMG_HEIGHT, 3)
x = x / 255

label = ["dog", "cat"]

#データの予測
pre = model.predict([x])[0]
idx = pre.argmax()
per = int(pre[idx] * 100)

draw(x)
print(f"この写真は{per}%の確率で{label[idx]}です")