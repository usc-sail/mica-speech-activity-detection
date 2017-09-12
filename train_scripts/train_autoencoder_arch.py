from __future__ import division
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation, Dropout
#from autoencoder_arch import *
from simple_autoenc import *
from read_scp import read_mat_scp_spect as rms
from pylab import *
from numpy import random as nprand
import os
import numpy as np
import tensorflow as tf
import keras.models
import threading
import time
from collections import Counter


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
##### DEFINITION OF PATHS AND GLOBAL VARIABLES

## Path Variables
proj_dir = '/proj/rajat/keras_model/gentle_aligned_data/'
TRAIN_SCP_FILE_SPEECH = proj_dir + 'feats/train/speech/shuf_labelled_spectrogram_feats.scp'
TRAIN_SCP_FILE_NON_SPEECH = proj_dir + 'feats/train/non_speech/shuf_spectrogram_feats.scp'
TEST_SCP_FILE_SP = proj_dir + 'feats/test/speech/shuf_labelled_spectrogram_feats.scp'
TEST_SCP_FILE_NS = proj_dir + 'feats/test/non_speech/shuf_spectrogram_feats.scp'
exp_dir = proj_dir + 'exp_data/exp7.4/'
log_dir = exp_dir + 'logs/'
os.system("mkdir -p %s" % (log_dir))
os.system("rm %s*" % (log_dir))

## Network Architecture
INP_DIM = (128,128)
BATCH_SIZE = 50
## Queueing-related Variables
FIRST_PASS_SP = 1
FIRST_PASS_NS = 1
NUM_THREADS = 8                ## Number of threads to be used
QUEUE_BATCH = 40              ## How many samples we put in a queue at a single enqueuing process
QUEUE_CAPACITY = 500         ## Capacity of queue
MIN_AFTER_DEQUEUE = 100      ## How big a buffer we will randomly sample from (bigger=>better shuffling but slow start-up and more memory used
QUEUE_BATCH_CAPACITY = 500
QUEUE_BATCH_MIN = 100


def initialize_training_variables():
    global MAX_EPOCHS, NUM_STEPS_TRAIN, TEST_SET_SIZE, MIN_LOSS, epoch_num, AVG_LOSS_EPOCH, batch_num, LR
    MAX_EPOCHS = 50
    NUM_STEPS_TRAIN = 2750
    TEST_SET_SIZE = 1000
    LR = 0.001
    MIN_LOSS = 1
    epoch_num=0
    AVG_LOSS_EPOCH = np.zeros(MAX_EPOCHS+1)
    batch_num=0

def generate_model_here():
    num_classes=32
    model = Sequential()
    # Block 1
    model.add(Conv2D(32, (3, 3), padding='valid', name='block1_conv1', input_shape=(INP_DIM[0],INP_DIM[1],1)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), padding='valid', name='block1_conv2'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    
#    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(64, (3, 3), padding='valid', name='block2_conv1'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    
    model.add(Conv2D(64, (3, 3), padding='valid', name='block2_conv2'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
 #   model.add(Dropout(0.25))
    
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    
    # Dense Layers
    model.add(Flatten(name='flatten'))
    model.add(Dense(512, activation='relu', name='fc1'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    
    model.add(Dense(512, activation='relu', name='fc2'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    
    model.add(Dense(num_classes, activation='sigmoid', name='predictions'))
    # Compile model
    #sgd = optimizers.SGD(lr=0.001, decay=1e-6)
    adam = keras.optimizers.Adam()
    # adam, adagrad
    model.compile(loss='binary_crossentropy', optimizer = adam, metrics=['accuracy'])
    #model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    # load weights
    return model

def save_model(avg_loss, model, epoch_num):
    global MIN_LOSS
    if(float(avg_loss)<MIN_LOSS):
        MIN_LOSS = avg_loss
        nnet_name = 'final_nnet_'+str(epoch_num)
        model.save(exp_dir+nnet_name)

def stopping_criterion(loss, epoch_num):
    if(epoch_num==1):
        return False
    elif((loss[epoch_num-1]-loss[epoch_num])<0.0001):
        return True
    return False
        
def generate_test_set(test_set_size):
    print("Generating Test Set")
    #subprocess.call('shuf -o %s %s'%(TEST_SCP_FILE,TEST_SCP_FILE), shell=True)
    gen_spc = rms(TEST_SCP_FILE_SP)
    gen_nsp = rms(TEST_SCP_FILE_NS)
    X_test = np.zeros((test_set_size,INP_DIM[0],INP_DIM[1]))
    count = 0
    speech_flag=1
    while True:
        if speech_flag==1:
            data = gen_spc.next()
        else:
            data = gen_nsp.next()
        for count in range(test_set_size):
            if(count < 0.8*(test_set_size)):
                X_test[count] = data[1]/255    
            else:
                speech_flag=0
                X_test[count] = data[1][:,:INP_DIM[1]]/255

        rand_keys = nprand.shuffle(range(test_set_size))
        X_test_shuf = X_test[rand_keys]
        #print("Generated Test Set of size %d" %test_set_size)
        #print("Fraction of Label 1 in test_set = %0.2f" %(sum(y_test)/len(y_test)))
        print("Test Set Generated")
        return X_test


## TF Queue stuff
input_feats_sp = tf.placeholder(tf.float32, shape = [QUEUE_BATCH, INP_DIM[0], INP_DIM[1]])
temp_var_sp = tf.placeholder(tf.int64, shape = [QUEUE_BATCH])
input_feats_ns = tf.placeholder(tf.float32, shape = [QUEUE_BATCH, INP_DIM[0], INP_DIM[1]])
temp_var_ns = tf.placeholder(tf.int64, shape = [QUEUE_BATCH])

queue_speech = tf.FIFOQueue(capacity = QUEUE_CAPACITY, dtypes=[tf.float32,tf.int64], shapes=[[INP_DIM[0],INP_DIM[1]],[]])
queue_non_speech = tf.FIFOQueue(capacity = QUEUE_CAPACITY, dtypes=[tf.float32,tf.int64], shapes=[[INP_DIM[0],INP_DIM[1]],[]])

enqueue_op_sp = queue_speech.enqueue_many([input_feats_sp, temp_var_sp])
dequeue_op_sp = queue_speech.dequeue_many(QUEUE_BATCH)

enqueue_op_ns = queue_non_speech.enqueue_many([input_feats_ns, temp_var_ns])
dequeue_op_ns = queue_non_speech.dequeue_many(QUEUE_BATCH)

## Batch generator using queue
data_batch_speech, target_batch_speech = tf.train.shuffle_batch(dequeue_op_sp, batch_size=25, capacity=QUEUE_BATCH_CAPACITY, min_after_dequeue=QUEUE_BATCH_MIN, enqueue_many=True)
data_batch_non_speech, target_batch_non_speech = tf.train.shuffle_batch(dequeue_op_ns, batch_size=25, capacity=QUEUE_BATCH_CAPACITY, min_after_dequeue=QUEUE_BATCH_MIN, enqueue_many=True)


### Feed in samples to queue
def enqueue_speech(sess):
    global FIRST_PASS_SP
    
    while True:
        if FIRST_PASS_SP==1:
            FIRST_PASS_SP = 0
            INPUT_DATA = rms(TRAIN_SCP_FILE_SPEECH)
        fts = np.zeros((QUEUE_BATCH,INP_DIM[0], INP_DIM[1]))
        for i in range(QUEUE_BATCH):
            try:
                data = INPUT_DATA.next()
            except:
                INPUT_DATA = rms(TRAIN_SCP_FILE_SPEECH)
                data = INPUT_DATA.next()
            fts[i] = data[1]/255
        sess.run(enqueue_op_sp, feed_dict={input_feats_sp: fts, temp_var_sp: np.zeros(QUEUE_BATCH)})

def enqueue_non_speech(sess):
    global FIRST_PASS_NS,COUNTNS
        
    while True:
        if FIRST_PASS_NS==1:
            FIRST_PASS_NS = 0
            INPUT_DATA = rms(TRAIN_SCP_FILE_NON_SPEECH)
            _,READ_DATA = INPUT_DATA.next()

        fts = np.zeros((QUEUE_BATCH,INP_DIM[0], INP_DIM[1]))
        start = 0 
        end = 0
        count = 0
        
        while count<QUEUE_BATCH:
            if READ_DATA.shape[1]>=INP_DIM[1]:
                fts[count] = READ_DATA[:,:INP_DIM[1]]/255
                READ_DATA = READ_DATA[:,INP_DIM[1]:]
                count+=1
            else:
                try:
                    _,READ_DATA = INPUT_DATA.next()
                except:
                    INPUT_DATA = rms(TRAIN_SCP_FILE_NON_SPEECH)
                    _,READ_DATA = INPUT_DATA.next()
                    
        sess.run(enqueue_op_ns, feed_dict={input_feats_ns: fts, temp_var_ns: np.zeros(QUEUE_BATCH)})
        
sess = tf.Session()
sess.run(tf.global_variables_initializer())
enqueue_thread_sp = threading.Thread(target=enqueue_speech, args=[sess])
enqueue_thread_sp.daemon = True
enqueue_thread_sp.start()
enqueue_thread_ns = threading.Thread(target=enqueue_non_speech, args=[sess])
enqueue_thread_ns.daemon = True
enqueue_thread_ns.start()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

######
######
## START - CODE TO DEBUG INPUT QUEUES

ct = 0
while ct<3:
    break
    ct+=1
    start = time.time()
    FIRST_PASS_SP=1
    FIRST_PASS_NS=1
#    logfile = open('log.txt','w')
    for i in range(3000):
        temp_num=i
        _,_ = sess.run([data_batch_speech, target_batch_speech])
        _,_ = sess.run([data_batch_non_speech, target_batch_speech])
        print('epoch-%d : batch-%d'%(ct,i))
    print('%f time elapsed'%(time.time()-start))     

    size = sess.run(queue_speech.size())
    sess.run(queue_speech.dequeue_many(size))
    size = sess.run(queue_non_speech.size())
    sess.run(queue_non_speech.dequeue_many(size))
#coord.request_stop()
#coord.join(threads)
#sess.run(queue.close(cancel_pending_enqueues=True))
#sess.close()
## END - CODE TO DEBUG INPUT QUEUES
######
######

### START TRAINING
initialize_training_variables()
## Compile model for training
model = Sequential()
m = generate_model()
for i in m:
    model.add(i)
sgd = optimizers.SGD(lr=LR, decay=1e-6)
adam = keras.optimizers.Adam()
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

## Generate Test Set
X_test= generate_test_set(TEST_SET_SIZE)
y_test = np.array([x.flatten() for x in X_test])
X_test = np.reshape(X_test,(TEST_SET_SIZE, INP_DIM[0], INP_DIM[1], 1))

while epoch_num<MAX_EPOCHS:
    epoch_num+=1
    batch_num=0                                                                 # Batch number of current batch in training
    loss_batch=0                                                                # Cumulative loss
    acc_batch=0                                                                 # Cumulative accuracy
    log_batch=log_dir+'log_batch_epoch_%d.txt'%(epoch_num)                      # Batch-wise log file
    log_epoch = open(log_dir + 'log_epoch.txt','a')                             # epoch-wise log file
    MIN_LOSS = 10
    FIRST_PASS_SP = 1
    FIRST_PASS_NS = 1
    start = time.time()
    ### Run a single epoch    
    while(batch_num<NUM_STEPS_TRAIN):
        batch_num+=1
        X_batch_speech,_ = sess.run([data_batch_speech, target_batch_speech])                 # get an input batch 
        X_batch_non_speech,_ = sess.run([data_batch_non_speech, target_batch_non_speech])                 # get an input batch 
        
        X_batch = np.concatenate((X_batch_speech, X_batch_non_speech), axis=0)
        rand_keys = nprand.shuffle(range(BATCH_SIZE))
        X_batch_shuf = X_batch[rand_keys]
     
        X_batch = np.reshape(X_batch_shuf,(BATCH_SIZE, INP_DIM[0], INP_DIM[1], 1))
        y_batch = np.array([x.flatten() for x in X_batch])
        metrics = model.fit(X_batch, y_batch, batch_size=BATCH_SIZE, epochs=1, verbose=2)         ## Train on a single batch of data
        acc_batch += float(format(metrics.history["acc"][0],'0.2f'))
        loss_batch += float(format(metrics.history["loss"][0],'0.2f'))
        ### Update log-file every 10 batches
        if(batch_num%10==0):        
            with open(log_batch,'a') as f:
                f.write("AVG LOSS : %0.4f, ACC : %0.2f after training %d batches\n" %((loss_batch/batch_num), (acc_batch/batch_num), batch_num))
            avg_loss_batch = loss_batch/batch_num
            save_model(avg_loss_batch, model, epoch_num)                        ## Save model if overall average loss has decreased
            ## Evaluate on test set every 100 batches
            if(batch_num%100==0):
                metrics_test = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=1)
                with open(log_batch,'a') as f:
                    f.write("TESTING - LOSS : %0.4f, ACC : %0.2f\n" %(metrics_test[0], metrics_test[1]))
    ### End of one epoch, empty queues and write epoch log
    AVG_LOSS_EPOCH[epoch_num] = loss_batch/batch_num
    log_epoch.write("EPOCH #%d TRAINING - AVG LOSS : %0.4f, AVG ACC : %0.2f\n\t\tTIME TAKEN - %f seconds\n" %(epoch_num, loss_batch/batch_num, acc_batch/batch_num, time.time()-start))
    log_epoch.close()
    if stopping_criterion(AVG_LOSS_EPOCH, epoch_num):                           # Stop training if loss has saturated
        print("Stopping Criterion Reached, stopping Training after epoch %d\n" %epoch_num)
        break
    ### Empty batch queue
    
    size = sess.run(queue_speech.size())
    sess.run(queue_speech.dequeue_many(size))
    size = sess.run(queue_non_speech.size())
    sess.run(queue_non_speech.dequeue_many(size))


coord.request_stop()
coord.join(threads)
sess.run(queue_speech.close(cancel_pending_enqueues=True))
sess.run(queue_non_speech.close(cancel_pending_enqueues=True))
sess.close()
