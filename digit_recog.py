# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:16:05 2020

@author: Pushkal Dwivedi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


#plotting a picture from x_train
g=plt.imshow(x_train[0][:,:,0], cmap=plt.get_cmap('gray'))



