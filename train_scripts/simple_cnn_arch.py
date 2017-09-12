from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation, Dropout

def generate_model():
    model = Sequential()
    # Block 1
    model.add(Conv2D(20, (5, 5), padding='valid', name='block1_conv1', input_shape=(INP_DIM[0],INP_DIM[1],1)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
    #model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(50, (5, 5), padding='valid', name='block2_conv1'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    #model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    #model.add(Dropout(0.25))
    
    model.add(Flatten(name='flatten'))
    model.add(Dense(512, activation='relu', name='fc1'))
    #model.add(BatchNormalization())
    #model.add(Activation("relu"))
    
    model.add(Dense(2, activation='softmax', name='fc2'))
    #model.add(BatchNormalization())
    #model.add(Activation("relu"))
    
    #model.add(Dense(num_classes, activation='sigmoid', name='predictions'))
    # Compile model
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)
    adam = keras.optimizers.Adam()
    # adam, adagrad
    model.compile(loss='binary_crossentropy', optimizer = sgd, metrics=['accuracy'])
    #model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    # load weights
    return model

