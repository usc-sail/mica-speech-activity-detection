from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation, Dropout

def generate_model():
    num_classes=128
    model = Sequential()
    # Block 1
    model.add(Conv2D(32, (5, 5), padding='valid', name='block1_conv1', input_shape=(INP_DIM[0],INP_DIM[1],1)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(32, (5, 5), padding='valid', name='block1_conv2'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(64, (5, 5), padding='valid', name='block2_conv1'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(Conv2D(64, (5, 5), padding='valid', name='block2_conv2'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    
    # block 3
   # model.add(Conv2D(128, (5, 5),  activation='relu', \
   #             padding='valid', name='block3_conv1'))
   # model.add(Conv2D(128, (5, 5), activation='relu', \
   #             padding='valid', name='block3_conv2'))
   # model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
   # 
    # Classification block
    
    model.add(Flatten(name='flatten'))
    model.add(Dense(1024, activation='relu', name='fc1'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(Dense(1024, activation='relu', name='fc2'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(Dense(num_classes, activation='sigmoid', name='predictions'))
    # Compile model
    #sgd = optimizers.SGD(lr=0.001, decay=1e-6)
    adam = keras.optimizers.Adam()
    # adam, adagrad
    model.compile(loss='binary_crossentropy', optimizer = adam, metrics=['accuracy'])
    #model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    # load weights
    return model

