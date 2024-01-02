"""
CNN 
学習済みモデルを検証する
"""

from os import pread
from IPython.core.display import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from PIL import Image
import matplotlib.pyplot as plt

#モデルの読み込み
model = model_from_json(open("dog_breed_model.json", "r").read())

model.load_weights("dog_breed_weight.h5")

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

label = ["affenpinscher", "Afghan_hound", "African_hunting_dog", "Airedale",
         "American_Staffordshire_terrier", "Appenzeller", "Australian_terrier", "basenji",
         "basset", "beagle", "Bedlington_terrier", "Bernese_mountain_dog",
         "black-and-tan_coonhound", "Blenheim_spaniel", "bloodhound", "bluetick",
         "Border_collie", "Border_terrier", "borzoi", "Boston_bull",
         "Bouvier_des_Flandres", "boxer", "Brabancon_griffon", "briard",
         "Brittany_spaniel", "bull_mastiff", "cairn", "Cardigan",
         "Chesapeake_Bay_retriever", "Chihuahua", "chow", "clumber",
         "cocker_spaniel", "collie", "curly-coated_retriever", "Dandie_Dinmont",
         "dhole", "dingo", "Doberman", "English_foxhound",
         "English_setter", "English_springer", "EntleBucher", "Eskimo_dog",
         "flat-coated_retriever", "French_bulldog", "German_shepherd", "German_short-haired_pointer",
         "giant_schnauzer", "golden_retriever", "Gordon_setter", "Great_Dane",
         "Great_Pyrenees", "Greater_Swiss_Mountain_dog", "groenendael", "Ibizan_hound",
         "Irish_setter", "Irish_terrier", "Irish_water_spaniel", "Irish_wolfhound",
         "Italian_greyhound", "Japanese_spaniel", "keeshond", "kelpie",
         "Kerry_blue_terrier", "komondor", "kuvasz", "Labrador_retriever",
         "Lakeland_terrier", "Leonberg", "Lhasa", "malamute",
         "malinois", "Maltese_dog", "Mexican_hairless", "miniature_pinscher",
         "miniature_poodle", "miniature_schnauzer", "Newfoundland", "Norfolk_terrier",
         "Norwegian_elkhound", "Norwich_terrier", "Old_English_sheepdog", "otterhound",
         "papillon", "Pekinese", "Pembroke", "Pomeranian",
         "pug", "redbone", "Rhodesian_ridgeback", "Rottweiler",
         "Saint_Bernard", "Saluki", "Samoyed", "schipperke",
         "Scotch_terrier", "Scottish_deerhound", "Sealyham_terrier", "Shetland_sheepdog",
         "Shih-Tzu", "Siberian_husky", "silky_terrier", "soft-coated_wheaten_terrier",
         "Staffordshire_bullterrier", "standard_poodle", "standard_schnauzer", "Sussex_spaniel",
         "Tibetan_mastiff", "Tibetan_terrier", "toy_poodle", "toy_terrier",
         "vizsla", "Walker_hound", "Weimaraner", "Welsh_springer_spaniel",
         "West_Highland_white_terrier", "whippet", "wire-haired_fox_terrier", "Yorkshire_terrier"]

#データの予測
pre = model.predict([x])[0]
idx = pre.argmax()
per = int(pre[idx] * 100)

draw(x)
print(f"この写真は{per}%の確率で{label[idx]}です")