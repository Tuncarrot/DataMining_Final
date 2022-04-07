from msilib.schema import Directory
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

# Directory where asl_dataset is stored, change this to your directory
DIR = "E:\School_Work\Winter_2022\DATA-CP421\Final_Project\\asl_dataset"

# Categories (stored as sub folders inside asl_dataset)
CAT_NUM = ['0','1','2','3','4','5','6','7','8','9']
CAT_LET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

CAT_ALL = ['0','1','2','3','4','5','6','7','8','9','a','b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Images sizes are 400x400, resize resolution to smaller file for faster computation
IMG_SIZE = 50

training_data = []

def create_training_data():
    for category in CAT_ALL:
        path = os.path.join(DIR, category)  # path to categories of numbers
        class_num = CAT_ALL.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) # Convert to grayscale, reduce file size by 3 (RGB->GS), colour is not essential in this case since we are tracking shape
                resize_img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))       # Resize
                training_data.append([resize_img_array, class_num])
            except Exception as e: # If any images are broken, pass
                pass


create_training_data()

print(len(training_data))

random.shuffle(training_data) # shuffle training data

X = [] # Feature Set
y = [] # Labels

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # param 1 is num of features (-1 is anything-goes), param 2 & 3 is img size, param 4 is 1 for grayscale
y = np.array(y)

pickle_out_X = open("X.pickle","wb")
pickle.dump(X, pickle_out_X)
pickle_out_X.close()

pickle_out_y = open("y.pickle","wb")
pickle.dump(y, pickle_out_y)
pickle_out_y.close()

# Export training files with pickle, so we dont have to keep recreating this