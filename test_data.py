import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

IMG_SIZE = 50

DIR = "C:\\Users\\jorda\\Desktop\\Files\\School\\Winter2022\\CP421\\Workspace\\DataMining_Final"

CAT_NUM = ['0','1','2','3','4','5','6','7','8','9']
CAT_LET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

CAT_ALL = ['0','1','2','3','4','5','6','7','8','9','a','b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

pickle_in_X = open("x_test.pickle", "rb") # Features
x_test = pickle.load(pickle_in_X)

pickle_in_y = open("y_test.pickle", "rb") # Labels
y_test = pickle.load(pickle_in_y)


x_test = tf.keras.utils.normalize(x_test, axis=1)  

asl_model = tf.keras.models.load_model('asl_model')

predictions = asl_model.predict(x_test)

i = 0
correct = 0

while i < 252:
    guess = np.argmax(predictions[i])

    if guess == y_test[i]:
        correct += 1

    print("Model:", guess, "  Actual:", y_test[i])
    i += 1

    

x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

score = asl_model.evaluate(x_test, y_test, verbose=False) 
asl_model.metrics_names
print('Test score: ', score[0])    #Loss on test
print('Test accuracy: ', score[1])