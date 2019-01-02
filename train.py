''' Training script for speech activity detection in movies
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
import keras
import tensorflow as tf
import numpy as np
from keras.layers import *
from keras.models import Model
from keras.backend.tensorflow_backend import set_session
import time
from SAD_parameters import *
from data_loader import Data

def get_session(gpu_fraction=0.333, num_cpus=8):
    if os.environ["CUDA_VISIBLE_DEVICES"] == '':
        config = tf.ConfigProto(device_count={"CPU":16})
    else:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)
    return tf.Session(config=config)

def ConvMPBlock(x, num_convs=2, fsize=32, kernel_size=3, pool_size=(2,2), strides=(2,2), BN=False, DO=False, MP=True):
    for i in range(num_convs):
       x = Conv2D(fsize, kernel_size, padding='same')(x)
       if BN:
           x = BatchNormalization()(x)
       if DO:
           x = Dropout(DO)(x)
       x = Activation('relu')(x)
    if MP:
        x = MaxPooling2D(pool_size=pool_size, strides=strides, padding='same')(x)
    return x

def FullyConnectedLayer(x, nodes=512, act='relu', BN=False, DO=False):
    x = Dense(nodes)(x)
    if BN:
        x = BatchNormalization()(x)
    if DO:
        x = Dropout(DO)(x)
    x = Activation(act)(x)
    return x

''' Define Speech activity detection model.
'''
def define_keras_model(input_shape=INPUT_SHAPE, optimizer='adam', loss='binary_crossentropy'):    
    fsize = 32
    td_dim = 256
    inp = Input(shape=input_shape)
    x = ConvMPBlock(inp, num_convs=2, fsize=fsize, BN=True)
    x = ConvMPBlock(x, num_convs=2, fsize=2*fsize, BN=True)
    x = ConvMPBlock(x, num_convs=3, fsize=4*fsize, BN=True)
    x = Reshape((x._keras_shape[1], x._keras_shape[2]*x._keras_shape[3]))(x)
    x = TimeDistributed(Dense(td_dim, activation='relu'))(x)
    x = GlobalAveragePooling1D()(x)
    x = FullyConnectedLayer(x, 128, BN=True)
    x = FullyConnectedLayer(x, 64, BN=True)
    x = FullyConnectedLayer(x, 2, 'softmax')
    model = Model(inp, x)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

''' Class defined to Monitor training and log metrics
    during training
'''
class Logger:
    def __init__(self, log_dir, num_epochs, num_steps):
        self.log_dir = log_dir
        if os.path.exists(log_dir):
            print("Please delete log directory manually and try again. Exiting...")
            exit()
        os.makedirs(log_dir)
        self.log_file = os.path.join(log_dir, 'training.log')
        self.model_file = os.path.join(log_dir, 'best_model.hdf5')
        self.train_metrics = {'acc':np.empty(num_steps), 'loss':np.empty(num_steps)}
        self.val_metrics = {'acc':np.empty(num_epochs), 'loss':np.empty(num_epochs)}
        self.start_time = time.time()
        self.best_loss = 1e6
        self.best_epoch = 0

    def log_batch_metrics(self, epoch, batch):
        log_epoch = os.path.join(self.log_dir, 'epoch_{}.log'.format(epoch+1))
        with open(log_epoch,'a') as fp_log:
            fp_log.write("Avg loss : %0.4f, accuracy : %0.2f after training %d batches\n"%(
                            np.mean(self.train_metrics['loss'][:batch]),
                            np.mean(self.train_metrics['acc'][:batch]), batch))
                            
    def log_epoch_metrics(self, epoch, val_metrics):
        self.val_metrics['loss'][epoch] = val_metrics[0]
        self.val_metrics['acc'][epoch] = val_metrics[1]
        with open(self.log_file,'a') as fp_log:
            fp_log.write("""EPOCH #%d TRAINING - avg loss : %0.4f, avg acc : %0.2f,
            VALIDATION - loss : %0.4f, acc : %0.2f \t\tTIME TAKEN - %0.2f minutes\n"""%(
            epoch+1, np.mean(self.train_metrics['loss']), 
            np.mean(self.train_metrics['acc']), self.val_metrics['loss'][epoch], 
            self.val_metrics['acc'][epoch], (time.time()-self.start_time)/60.0))
        print("Validation set - loss : %0.4f, acc : %0.2f after epoch %d"%(
                self.val_metrics['loss'][epoch], self.val_metrics['acc'][epoch], epoch+1))
    
    def log_test_metrics(self, test_metrics):
        with open(self.log_file, 'a') as fp_log:
            fp_log.write("TESTING - loss : %0.4f, acc : %0.2f"%(
                test_metrics[0], test_metrics[1]))
        print("Test set - loss : %0.4f, acc : %0.2f"%(test_metrics[0], test_metrics[1]))

    def save_model(self, epoch):
        if self.val_metrics['loss'][epoch] < self.best_loss:
            self.best_loss = self.val_metrics['loss'][epoch]
            self.best_epoch = epoch
            model.save(self.model_file)
            print("Model saved after epoch %d"%(epoch+1))

    def early_stopping(self, epoch, patience):
        if epoch - self.best_epoch > patience:
            print("Early stopping criterion met, stopping training...")
            return True
        return False

''' Test the model on external test set after training
'''
def test_model(data_obj, model, sess):
    test_acc = []
    test_loss = []
    while 1:
        try:
            X_test, y_test = sess.run([data_obj.X_test, data_obj.y_test])
            metrics = model.test_on_batch(X_test, y_test)
            test_loss.append(metrics[0])
            test_acc.append(metrics[1])
        except:
            break
    return np.mean(test_loss), np.mean(test_acc)

''' Train model with default parameters defined in
    SAD_parameters.py
'''
def train_model(model, sess, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, num_steps=NUM_STEPS, patience=PATIENCE, log_dir=LOG_DIR, log_freq=LOG_FREQ):
    data_obj = Data()
    X_val, y_val = sess.run([data_obj.X_val, data_obj.y_val])
    logger = Logger(log_dir=log_dir, num_epochs=num_epochs, num_steps=num_steps)

    for epoch in range(num_epochs):
        #initialize per_epoch variables
        print("\nBeginning epoch %d"%(epoch+1))
        logger.start_time = time.time()
        for batch in range(num_steps):
            X_batch, y_batch = sess.run([data_obj.X_batch, data_obj.y_batch])
            # Train on single batch
            metrics = model.train_on_batch(X_batch, y_batch)
            logger.train_metrics['loss'][batch] = metrics[0]
            logger.train_metrics['acc'][batch] = metrics[1]

            if batch % log_freq == 0 and batch!=0:  # Log training metrics every 'log_freq' batches
                logger.log_batch_metrics(epoch, batch)
        
        val_metrics = model.evaluate(X_val, y_val)
        logger.log_epoch_metrics(epoch, val_metrics)
        logger.save_model(epoch)
        if logger.early_stopping(epoch, patience): 
            break                           ## Stop training if loss is not decreasing

    test_metrics = test_model(data_obj=data_obj, model=model, sess=sess)    # Evaluate performance on external test set
    logger.log_test_metrics(test_metrics)
    sess.close()

if __name__ == '__main__':
    sess = get_session(gpu_fraction=GPU_FRAC)
    set_session(sess)
    model = define_keras_model(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE))
    train_model(model=model, sess=sess)

