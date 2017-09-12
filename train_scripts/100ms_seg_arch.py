from __future__ import division
from sklearn.metrics import hamming_loss
from keras.models import Sequential
from functools import partial
from keras import optimizers
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation, Dropout,GlobalAveragePooling2D
import keras.models
#from keras import backend as K
from keras.layers.core import K
import numpy as np

def generate_model():
    model = Sequential()
    # Block 1
    model.add(Conv2D(32, (3, 3), padding='same', name='block1_conv1', input_shape=(128,10,1)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), padding='same', name='block1_conv2'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    
#    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(64, (3, 3), padding='same', name='block2_conv1'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    
    model.add(Conv2D(64, (3, 3), padding='same', name='block2_conv2'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
 #   model.add(Dropout(0.25))
    
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    
    # Block 3
    model.add(Conv2D(128, (3, 3), padding='same', name='block3_conv1'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    
    model.add(Conv2D(128, (3, 3), padding='same', name='block3_conv2'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
 #   model.add(Dropout(0.25))
    
    # Dense Layers
    #model.add(Dense(512, activation='relu', name='fc1'))
    model.add(Flatten(name='flatten'))
    #model.add(BatchNormalization())
    
    model.add(Dense(512, activation='relu', name='fc2'))
    model.add(BatchNormalization())
    
    #model.add(Activation("sigmoid"))
    model.add(Dense(2, activation='softmax', name='predictions'))
    # Compile model
    #sgd = optimizers.SGD(lr=0.001, decay=1e-6)
    adam = keras.optimizers.Adam()
    # adam, adagrad
    model.compile(loss='binary_crossentropy', optimizer = adam, metrics=['accuracy'])
    #model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    # load weights
    return model

