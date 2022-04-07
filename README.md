# DataMining_Final

This project contains two python scripts. Both are used to create and train a model to recognize various sign language for the numbers 0-9 and letters A-Z.

## Requirements

PIP install the following packages to get the scripts running

pip install --upgrade tensorflow
pip install numpy
pip install matplotlib
pip install opencv-python

Additionally, download and save the following sign language dataset

https://www.kaggle.com/datasets/ayuraj/asl-dataset

## create_training_data.py

This file creates the training data for the model to study from. When using, replace the directory with where you saved your dataset

By default, the script will create a dataset with all the categories (0-9 and A-Z).

The training data will be saved as pickle files to be consumed by the model.

## train_data.py

This file takes in the pickle files generated by the previous file and trains a neural network with them.
