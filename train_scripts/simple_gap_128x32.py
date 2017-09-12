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

#- start weighted xent here
def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:,:, 0])
    y_pred_max = K.max(y_pred, axis=-1)
    y_pred_max = K.expand_dims(y_pred_max, axis=-1)
    y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
    for c_p, c_t in itertools.product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:,:, c_p] * y_true[:,:, c_t])
    return K.categorical_crossentropy(y_pred, y_true) * final_mask


def hamm_loss(y_true, y_pred):
    y_pred_clss = K.mean(K.not_equal(K.round(y_pred),y_true))
    loss = y_pred_clss
    return K.variable(loss)

ncce = partial(hamm_loss)
ncce.__name__ = 'hamm_loss'

def generate_model_simple_128x32():
    num_classes=32
    model = Sequential()
    # Block 1
    model.add(Conv2D(32, (3, 3), padding='same', name='block1_conv1', input_shape=(128,32,1)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), padding='same', name='block1_conv2'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    
#    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(32, (3, 3), padding='same', name='block2_conv1'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    
    model.add(Conv2D(32, (3, 3), padding='same', name='block2_conv2'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
 #   model.add(Dropout(0.25))
    
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    
    # Block 3
    model.add(Conv2D(32, (3, 3), padding='same', name='block3_conv1'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    
    model.add(Conv2D(32, (3, 3), padding='same', name='block3_conv2'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
 #   model.add(Dropout(0.25))
    
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    # Dense Layers
    #model.add(Dense(512, activation='relu', name='fc1'))
    #model.add(Flatten(name='flatten'))
    #model.add(BatchNormalization())
    
    #model.add(Dense(512, activation='relu', name='fc2'))
    #model.add(BatchNormalization())
    
    model.add(GlobalAveragePooling2D(data_format='channels_last'))
    model.add(Activation("sigmoid"))
    #model.add(Dense(num_classes, activation='sigmoid', name='predictions'))
    # Compile model
    #sgd = optimizers.SGD(lr=0.001, decay=1e-6)
    adam = keras.optimizers.Adam()
    # adam, adagrad
    model.compile(loss=ncce, optimizer = adam, metrics=['accuracy'])
    #model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    # load weights
    return model

