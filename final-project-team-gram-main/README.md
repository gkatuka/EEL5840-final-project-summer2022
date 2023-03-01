# Final Project

## Training Data

To use train.ipynb, you will need a data.npy file and a labels.npy file. You can either name your files these,
or you can modify the name of the variables "X_train_full" and "t_train_full". Then you may call the train() function and it will generate and return a model. We have already run this function with our dataset and saved the model as TF_model.h5.

## Testing Data

To use test.ipynb, you will need a data.npy file and a labels.npy file. These files contain the data you want to
use to test the model we have. You need to make sure these files are in the same format as the training data from the project, you can then call test(X_full, t_full) passing in the data and labels respectively. This function will return a numpy array with the predicted lables as well as print the predictions and accuracy. The workspace also will need our TF_model.h5 file. When you run the jupyter notebook,
it will predict the X_full data using the TF_model and compare against t_full.

## Extra credit

To use hard_test.ipynb, you will need a data.npy file and a labels.npy file. These files contain the data you want to
use to test the model we have. You need to make sure these files are in the same format as the training data from the project, you can then call hard_test(X_full, t_full) passing in the data and labels respectively. This function will return a numpy array with the predicted lables as well as print the predictions and accuracy. The workspace also will need our TF_model.h5 file. When you run the jupyter notebook,
it will predict the X_full data using the TF_model and compare against t_full.

This script assumes the unknown label will be labeled as -1.

## Dependencies

numpy, matplotlib, tensorflow, keras, sklearn

import numpy as np<br>
import numpy.random as npr<br>
import matplotlib.pyplot as plt<br>
import tensorflow as tf<br>
from tensorflow import keras<br>
from sklearn.metrics import classification_report<br>
from sklearn.model_selection import train_test_split<br>
<br>

These functions assume the dataset contains at least one of each class.
