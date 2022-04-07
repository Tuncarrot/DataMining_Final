import tensorflow as tf
import pickle
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

pickle_in_X = open("X.pickle", "rb") # Features
X = pickle.load(pickle_in_X)

pickle_in_y = open("y.pickle", "rb") # Labels
y = pickle.load(pickle_in_y)


#X = X/255.0 # scale data to normalize, 255 for pixel data
X = tf.keras.utils.normalize(X, axis=1)                             # normalize values inside array (decrease range from 0-255 to 0-1)

model = Sequential()                                                # Initalize sequential model
model.add(Flatten())                                                # Flatten from multi-dimensional array to 1D array
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))        # Add dense hidden layer with 128 neurons and ReLU act. function
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))        # Add dense hidden layer with 128 neurons and ReLU act. function
model.add(tf.keras.layers.Dense(36, activation=tf.nn.softmax))      # Add dense output layer with 36 neruons (10 numbers + 26 letters) and softmax function

model.compile(loss="sparse_categorical_crossentropy",          # Categorical_crossentroy because its non-binary
            optimizer ="adam",
            metrics=['accuracy'])

model.fit(X, y, epochs=15)    # Feed in batch_num images at once
