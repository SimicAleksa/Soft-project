import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications import InceptionV3
from tensorflow import keras
from keras import layers as tfl
import os
import cv2
from sklearn import preprocessing
from pathlib import Path
from skimage import io, transform
from PIL import Image


# function to extract y labels from base file path
def find_label_map():
    return {
        "1L":0,
        "1R":1,
        "2L":2,
        "2R":3,
        "3L":4,
        "3R": 5,
        "4L": 6,
        "4R": 7,
        "5L": 8,
        "5R": 9,
        "sL": 10,
        "sR": 11,
        "aL": 12,
        "oR": 13
    }


def find_reverse_label_map():
    return {
        0: "1",
        1: "6",
        2: "2",
        3: "7",
        4: "3",
        5: "8",
        6: "4",
        7: "9",
        8: "5",
        9: "0",
        10: "+",
        11: "-",
        12: "*",
        13: "/"
    }


def extract_label(base):
    path = []
    label = []
    label_map = find_label_map()
    lista_suffle = os.listdir(base)
    random.shuffle(lista_suffle)
    for filename in lista_suffle:
        if (len(filename.split('.')[0]) < 18):
            # specijalni karakteri
            char = filename.split('(')[0]
            label.append(label_map[char])  # split on dot and read the preceding 2 literals
        else:
            char = filename.split('.')[0][-2:]
            label.append(label_map[char])  # split on dot and read the preceding 2 literals
        path.append(base + filename)
    print("Labele ovdje su ", label)
    return path, label


train_base = "treningMix/"
test_base = "testMix/"


train_set_path, train_set_label = extract_label(train_base)
test_set_path, test_set_label = extract_label(test_base)

nase_labele = list(np.unique(train_set_label))

print("Ukupno za treniranje: ", len(train_set_path))
print("Ukupno za test: ", len(test_set_path))


# function to split feature data into train and test
def feature_data_split(path):
    feature_set = []
    for img in path:
        img_read = io.imread(img)
        # Most images are already of size (128,128) but it is always better to ensure they all are
        img_read = transform.resize(img_read, (128, 128, 1), mode='constant')
        feature_set.append(img_read)
    return feature_set



X_test = feature_data_split(test_set_path)
X_train = feature_data_split(train_set_path)


X_train = np.array(X_train)
X_test = np.array(X_test)


print(X_train.shape, " je X_train.shape")
print(X_test.shape, " je X_test.shape")

print("Train Label set unique values: ", list(np.unique(train_set_label)))
print("Test  Label set unique values: ", list(np.unique(test_set_label)))
if len(np.unique(train_set_label)) == len(np.unique(test_set_label)):
    print("Number of unique classes: ", len(np.unique(train_set_label)))
    num_classes = len(np.unique(train_set_label))


def nassaModel():
    model = keras.Sequential()

    #### Input Layer ####
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                     activation='relu', input_shape=(128, 128, 1)))

    #### Convolutional Layers ####
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))  # Pooling
    model.add(keras.layers.Dropout(0.2))  # Dropout

    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Conv2D(512, (5, 5), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(512, (5, 5), activation='relu'))
    model.add(keras.layers.MaxPooling2D((4, 4)))
    model.add(keras.layers.Dropout(0.2))

    #### Fully-Connected Layer ####
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(len(train_set_label), activation='softmax'))

    return model

# conv_model = nassaModel()
# conv_model.compile(optimizer='adam',
#                    loss='sparse_categorical_crossentropy',
#                    metrics=['accuracy'])
# conv_model.summary()
# my_model = conv_model.fit(np.array(X_train,float),np.array(train_set_label,float), epochs=10, validation_data=(np.array(X_test,float),np.array(test_set_label,float)))
#
#
#
# conv_model.save('modelFinalni10Pravi')

conv_model = tf.keras.models.load_model('modelFinalni15')

# loss,accuracy = conv_model.evaluate(np.array(X_test,float), np.array(test_set_label,float))
# print("LOSS JE",loss)
# print("ACC JE",accuracy)

videoPath = "necaVideo32.mp4"
cap = cv2.VideoCapture(videoPath)  # dobijamo pristup nasoj web kameri
frame_num = 0
cap.set(1, frame_num)
reverse_label = find_reverse_label_map()
racunaj= ""
while True:  # imamo loop frejmova dobijenih od web kamere
    frame_num += 1
    grabbed, frame = cap.read()
    # frejmovi nisu formata 250x250px, zato ih rucno podesavamo
    if grabbed:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.resize(frame_gray, (128, 128))

        cv2.imshow('Web cam', frame_gray)  # prikaz prozora live feed-as
        # za kreiranje positive i anchor slika koristimo lib uuid - koja obezbedjuje jedinstvene nazive
        if cv2.waitKey(0) & 0XFF == ord('a'):  # za pravljenje chor slika
            # kreiramo jedinstvenu putanju na kojoj cemo da cuvamo sliku
            imgname = os.path.join("calc", 'slika.jpg')
            # samo cuvanje slike
            cv2.imwrite(imgname, frame_gray)
            prediction = conv_model.predict(np.array([frame_gray]) / 255)
            index = np.argmax(prediction)
            print(prediction[0], " je predikcija od 0")
            max = 0
            maxIndex = 0
            for i in range(len(prediction[0])):
                if prediction[0][i] > max:
                    max = prediction[0][i]
                    maxIndex = i
            # print(f'Prediction is {reverse_label[nase_labele[index]]}')
            cc = reverse_label[nase_labele[index]]
            racunaj+=cc
            print("Trenutni izraz je: ",racunaj)

        if cv2.waitKey(0) & 0XFF == ord('b'):  # za pravljenje anchor slika
            pass
        if cv2.waitKey(0) & 0XFF == ord('x'):  # za pravljenje anchor slika
            cap.release()  # prekidamo konekciju sa web kamerom
            cv2.destroyAllWindows()  # zatvaramo prozor live feed-a
            break
    else:
        break

print("Kalkulator vraca; ", eval(racunaj))


