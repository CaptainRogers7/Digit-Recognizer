# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:16:05 2020

@author: Pushkal Dwivedi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import RMSprop,Adam
data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

y_train = np.array(data_train['label'])

x_train= np.array(data_train.drop(columns="label"))
x_test = np.array(data_test)

#standardizing the dataset
x_train = x_train/255
x_test = x_test/255
x_test=x_test.reshape(x_test.shape[0],28,28,1)
x_train=x_train.reshape(x_train.shape[0],28,28,1)
y_train.reshape(y_train.shape[0],-1).T

#since this is a multiclass classification problem we need to one hot encode our labels
y_tr = to_categorical(y_train, num_classes=10)

#plotting a picture from x_train
g=plt.imshow(x_train[27][:,:,0], cmap=plt.get_cmap('gray'))

#initialising the cnn model as sequential
classifier = Sequential()

#adding the first layer
classifier.add(Conv2D(filters=22,kernel_size=(5,5),padding='Same',activation='relu',input_shape=(28,28,1)))
#adding the pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))


# Adding a second convolutional layer
classifier.add(Conv2D(filters=32,kernel_size=(4,4),padding='Same',activation='relu',input_shape=(28,28,1)))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#adding dropout for regularization
classifier.add(Dropout(0.20))

#full connection
classifier.add(Flatten())
classifier.add(Dense(256, activation = "relu"))
classifier.add(Dropout(0.5))
#using softmax as it is multi class classification
classifier.add(Dense(10, activation = "softmax"))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#fitting the data
classifier.fit(x_train, y_train, batch_size = 80, epochs = 25)

#preicting
y_pred = classifier.predict(x_test)

#highest accuracy achieved- 99.69


#accuracy after applying hyerparameter tuning
import hyperpara_tuning as ht
best_parameters,best_accuracy = ht.tune_parameter(x_train,y_train)