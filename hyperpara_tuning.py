# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 04:00:16 2020

@author: Pushkal Dwivedi
"""
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import RMSprop,Adam
def tune_parameter(x_train,y_train):
    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import GridSearchCV
    from keras.models import Sequential
    from keras.layers import Dense
    def build_classifier(optimizer):
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
        return classifier
        
    classifier = KerasClassifier(build_fn = build_classifier)
    parameters = {'batch_size': [70, 90],
                  'epochs': [20, 30],
                  'optimizer': ['adam', 'rmsprop']}
    grid_search = GridSearchCV(estimator = classifier,
                               param_grid = parameters,
                               scoring = 'accuracy',
                               cv = 10)
    grid_search = grid_search.fit(x_train, y_train)
    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_
    return best_parameters,best_accuracy