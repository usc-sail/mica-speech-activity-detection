import keras
from keras.layers import Conv2D, Activation, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, ZeroPadding2D, Conv2DTranspose, Reshape, Permute, Dense, Flatten

def generate_model():
    kernel = 3
    filter_size = 2
    pad = 1
    pool_size = 2
    stride = (2,2)
    data_shape = 128*128
    return [
        ### Encoder network
        Conv2D(filter_size, kernel, padding='same', input_shape=(128,128,1)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size), strides=stride),

        Conv2D(4, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size), strides=stride),

        Conv2D(8, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        Flatten(),
        Dense(1024, activation='relu'),


        ### Decoder network
        #UpSampling2D(size=(pool_size,pool_size)),
        # ZeroPadding2D(padding=(pad,pad)),
        Reshape((32,32,1)),
        Conv2DTranspose(8, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),


        UpSampling2D(size=(pool_size,pool_size)),
        #ZeroPadding2D(padding=(pad,pad)),
        Conv2DTranspose(4, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(pool_size,pool_size)),
        # ZeroPadding2D(padding=(pad,pad)),
        Conv2DTranspose(filter_size, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        ### Target Layer
        Conv2D(1, kernel, padding='same', activation='sigmoid'),
        Flatten()
    ]

