"""
CNN
画像データをTensorflowで学習できるように変換する
"""

import numpy as np
import glob
import tensorflow as tf
from PIL import Image
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_HEIGHT = 224
IMG_WIDTH = 224

#1. データの準備
def prepare_data():
  #データのダウンロード
  max_photo = 200

  x = []
  t = []

  def read_image(path, label):
    files = glob.glob(path + "/*jpg")
    num = 0
    for f in files:
      if num >= max_photo: break
      num += 1
      img = Image.open(f)
      img = img.convert("RGB")
      img = img.resize((IMG_HEIGHT, IMG_WIDTH))
      img = np.asarray(img)
      x.append(img)
      t.append(label)

  labels = ["affenpinscher", "Afghan_hound", "African_hunting_dog", "Airedale",
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

  for index, label in enumerate(labels):
    read_image("drive/My Drive/Colab Notebooks/Images/Dog_Breed/"+label, index)

  x = np.array(x)
  t = np.array(t)

  #データをtrain用とval用に分ける
  x = (x.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 3)/255).astype(np.float32)

  mean = np.mean(x)
  std = np.std(x)

  x = (x - mean)/(std + 1e-7)

  t = to_categorical(t) #One-Hot　表現に

  x_train, x_val, t_train, t_val = train_test_split(x, t, test_size = 0.2)

  np.savez("drive/MyDrive/Colab Notebooks/Images/train_dogs_data", x = x_train, y = t_train )
  np.savez("drive/MyDrive/Colab Notebooks/Images/val_dogs_data", x = x_val, y = t_val)

prepare_data()