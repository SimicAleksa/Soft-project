#  import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers as tfl
import os
import cv2
from sklearn import preprocessing
from pathlib import Path
from skimage import io, transform
from PIL import Image


# function to extract y labels from base file path
def extract_label(base):
    path = []
    label = []
    for filename in os.listdir(base):
        if (len(filename.split('.')[0]) < 18):
            label.append(filename.split('(')[0])  # split on dot and read the preceding 2 literals
        else:
            label.append(filename.split('.')[0][-2:])  # split on dot and read the preceding 2 literals
        path.append(base + filename)
    return path, label


# base file paths for train and test folders
train_base = "trening/"
test_base = "test/"

# list variables holding file paths and y labels for train and test sets
train_set_path, train_set_label = extract_label(train_base)
test_set_path, test_set_label = extract_label(test_base)

print("Number of training set examples: ", len(train_set_path))
print("Number of test set examples: ", len(test_set_path))


# function to split feature data into train and test
def feature_data_split(path):
    feature_set = []
    for img in path:
        img_read = io.imread(img)
        # Most images are already of size (128,128) but it is always better to ensure they all are
        img_read = transform.resize(img_read, (128, 128, 1), mode='constant')
        feature_set.append(img_read)
    return feature_set


# read images for train and test set
X_test = feature_data_split(test_set_path)
X_train = feature_data_split(train_set_path)

# convert lists into np arrays to facilitate modelling
X_train = np.array(X_train)
X_test = np.array(X_test)

# check shapes
print(X_train.shape)
print(X_test.shape)

# check unique values in train and test sets
print("Train Label set unique values: ", list(np.unique(train_set_label)))
print("Test  Label set unique values: ", list(np.unique(test_set_label)))
if len(np.unique(train_set_label)) == len(np.unique(test_set_label)):
    print("Number of unique classes: ", len(np.unique(train_set_label)))
    num_classes = len(np.unique(train_set_label))

# apply label encoder to the y train and test sets
label_encoder = preprocessing.LabelEncoder()
y_train = label_encoder.fit_transform(train_set_label)
y_test = label_encoder.fit_transform(test_set_label)

print("Label Encoded Train Label set unique values: ", np.unique(y_train))
print("Label Encoded Test  Label set unique values: ", np.unique(y_test))
if len(np.unique(y_train)) == len(np.unique(y_test)):
    print("Number of unique classes: ", len(np.unique(y_train)))
    num_classes = len(np.unique(y_train))

# one hot encode y label train and test sets
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(y_train.shape)
print(y_test.shape)


# def convolutional_model(input_shape):
#     input_img = tf.keras.Input(shape=input_shape)
#     conv_layer1 = keras.layers.Conv2D(filters=8, kernel_size=4, strides=1, padding='same')(input_img)
#     activation1 = keras.layers.ReLU()(conv_layer1)
#     pool_layer1 = keras.layers.MaxPool2D(pool_size=8, strides=8, padding='same')(activation1)
#     conv_layer2 = keras.layers.Conv2D(filters=16, kernel_size=2, strides=1, padding='same')(pool_layer1)
#     activation2 = keras.layers.ReLU()(conv_layer2)
#     pool_layer2 = keras.layers.MaxPool2D(pool_size=4, strides=4, padding='same')(activation2)
#     flatten = keras.layers.Flatten()(pool_layer2)
#     outputs = keras.layers.Dense(units=num_classes, activation='softmax')(flatten)
#     model = tf.keras.Model(inputs=input_img, outputs=outputs)
#     return model
#
#
# # Compile ConvNet model
# conv_model = convolutional_model((128, 128, 1))
# conv_model.compile(optimizer='adam',
#                    loss='categorical_crossentropy',
#                    metrics=['accuracy'])
# conv_model.summary()
#
# # train ConvNet Model
# train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(128)
# test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(128)
#
# my_model = conv_model.fit(train_dataset, epochs=30, validation_data=test_dataset)
#
# # Returns the loss value & metrics values for the model in test mode
# conv_model.evaluate(X_test, y_test)
#
# conv_model.save('cuvaniModel')

conv_model = tf.keras.models.load_model('cuvaniModel')
# sample_predictions = conv_model.predict(X_test[:100])
# sample_predictions[10:20]
# fig, axs = plt.subplots(2, 5, figsize=[24, 12])
#
# count = 14
# for i in range(2):
#     for j in range(5):
#         img = cv2.imread(test_set_path[count])
#         results = np.argsort(sample_predictions[count])[::-1]
#         labels = label_encoder.inverse_transform(results)  # get the label names using inverse transform
#         axs[i][j].imshow(img)
#         axs[i][j].set_title(labels[0], fontsize=20)
#         axs[i][j].axis('off')
#         count += 8
#
# plt.suptitle("Sample Predictions", fontsize=24)
# plt.show()


# ////////////////////////////////////////////////////////////////////////
def process_video(video_path, classifier):
    # procesiranje jednog videa
    sum_of_nums = 0

    # ucitavanje videa
    frame_num = 0
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_num)  # indeksiranje frejmova

    # analiza videa frejm po frejm
    while True:
        frame_num += 1
        grabbed, frame = cap.read()
        frame = transform.resize(frame, (128, 128, 1))
        print(frame.shape)
        # ako frejm nije zahvacen
        if not grabbed:
            break
        res = classifier.predict(frame)
        print(res)
        cv2.imshow('image', frame)
        cv2.waitKey(0)

    cap.release()
    return sum_of_nums


getRes = process_video("sakaSoftVideo.mp4", conv_model)
print(getRes)
