from __future__ import division
from keras.models import Sequential
from keras import backend as K
from keras import optimizers
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, Reshape, LSTM
from read_scp import read_mat_scp_spect as rms
from keras.models import load_model
from simple_gap_128x32 import *
from numpy import random as nprand
from scipy import misc
import os
import numpy as np
import tensorflow as tf
import keras.models
import threading
import random
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
model_path = proj_dir + 'exp_data/exp7.3/'
init_model = load_model(model_path+'final_nnet_4')
exp_dir = proj_dir + 'exp_data/exp8.1/'
log_dir = exp_dir + 'logs/'
os.system("mkdir -p %s" % (log_dir))
os.system("rm %s*" % (log_dir))

## Network Architecture
INP_DIM = (128,128)
BATCH_SIZE = 40
## Queueing-related Variables
FIRST_PASS_SP = 1
FIRST_PASS_NS = 1
NUM_THREADS = 8                ## Number of threads to be used
QUEUE_BATCH = 40              ## How many samples we put in a queue at a single enqueuing process
QUEUE_CAPACITY = 500         ## Capacity of queue
MIN_AFTER_DEQUEUE = 100      ## How big a buffer we will randomly sample from (bigger=>better shuffling but slow start-up and more memory used
QUEUE_BATCH_CAPACITY = 500
QUEUE_BATCH_MIN = 100

def maximum(arr):
    k = [abs(x) for x in arr]
    return max(k)

def initialize_training_variables():
    global MAX_EPOCHS, NUM_STEPS_TRAIN, TEST_SET_SIZE, MIN_LOSS, epoch_num, AVG_LOSS_EPOCH, batch_num
    MAX_EPOCHS = 50
    NUM_STEPS_TRAIN = 2700
    TEST_SET_SIZE = 1000
    MIN_LOSS = 1
    epoch_num=0
    AVG_LOSS_EPOCH = np.zeros(MAX_EPOCHS+1)
    batch_num=0

def modify_autoenc_model(init_model):
    k=''
    while 'dense' not in k:
        k = init_model.layers.pop().name
        init_model.outputs = [init_model.layers[-1].output]
        init_model.output_layers = [init_model.layers[-1]]
        init_model.layers[-1].outbound_nodes = []

    for i in range(8):
        init_model.layers[i].trainable=False
    
    init_model.add(Reshape((128,64)))
    init_model.add(LSTM(2, return_sequences='True', input_shape=(128,64), name='lstm'))
    init_model.add(Activation('softmax', name='predictions'))
    eodel.add(GlobalAveragePooling2D(data_format='channels_last'))
    #sgd = optimizers.SGD(lr=0.001, decay=1e-6)
    adam = keras.optimizers.Adam()
    init_model.compile(loss='binary_crossentropy', optimizer = adam, metrics=['accuracy'])
    return init_model

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
    #subprocess.call('shuf -o %s %s'%(TEST_SCP_FILE,TEST_SCP_FILE), shell=True)
    gen_spc = rms(TEST_SCP_FILE_SP)
    gen_nsp = rms(TEST_SCP_FILE_NS)
    X_test = np.zeros((test_set_size,INP_DIM[0],INP_DIM[1]))
    y_test = np.zeros((test_set_size, INP_DIM[1]))
    speech_flag=1
    while True:
        if speech_flag==1:
            data = gen_spc.next()
        else:
            data = gen_nsp.next()
        for count in range(test_set_size):
            if(count < 0.5*(test_set_size)):
                X_test[count] = data[1]/255
                y_test[count] = data[2]
            else:
                speech_flag=0
                X_test[count] = data[1][:,:INP_DIM[1]]/255
                y_test[count] = np.zeros(INP_DIM[1])

        #print("Generated Test Set of size %d" %test_set_size)
        #print("Fraction of Label 1 in test_set = %0.2f" %(sum(y_test)/len(y_test)))
        return X_test, y_test


## TF Queue stuff
input_feats_sp = tf.placeholder(tf.float32, shape = [QUEUE_BATCH, INP_DIM[0], INP_DIM[1]])
input_labels_sp = tf.placeholder(tf.int64, shape = [QUEUE_BATCH, INP_DIM[1]])
input_feats_ns = tf.placeholder(tf.float32, shape = [QUEUE_BATCH, INP_DIM[0], INP_DIM[1]])
input_labels_ns = tf.placeholder(tf.int64, shape = [QUEUE_BATCH, INP_DIM[1]])

queue_speech = tf.FIFOQueue(capacity = QUEUE_CAPACITY, dtypes=[tf.float32, tf.int64], shapes=[[INP_DIM[0],INP_DIM[1]],[INP_DIM[1]]])
queue_non_speech = tf.FIFOQueue(capacity = QUEUE_CAPACITY, dtypes=[tf.float32, tf.int64], shapes=[[INP_DIM[0],INP_DIM[1]],[INP_DIM[1]]])

enqueue_op_sp = queue_speech.enqueue_many([input_feats_sp, input_labels_sp])
dequeue_op_sp = queue_speech.dequeue_many(QUEUE_BATCH)

enqueue_op_ns = queue_non_speech.enqueue_many([input_feats_ns, input_labels_ns])
dequeue_op_ns = queue_non_speech.dequeue_many(QUEUE_BATCH)

## Batch generator using queue
data_batch_speech, target_batch_speech = tf.train.shuffle_batch(dequeue_op_sp, batch_size=25, capacity=QUEUE_BATCH_CAPACITY, min_after_dequeue=QUEUE_BATCH_MIN, enqueue_many=True)
data_batch_non_speech, target_batch_non_speech = tf.train.shuffle_batch(dequeue_op_ns, batch_size=15, capacity=QUEUE_BATCH_CAPACITY, min_after_dequeue=QUEUE_BATCH_MIN, enqueue_many=True)


### Feed in samples to queue
def enqueue_speech(sess):
    global FIRST_PASS_SP
    
    while True:
        if FIRST_PASS_SP==1:
            FIRST_PASS_SP = 0
            INPUT_DATA = rms(TRAIN_SCP_FILE_SPEECH)
        fts = np.zeros((QUEUE_BATCH,INP_DIM[0], INP_DIM[1]))
        tgts =  np.zeros((QUEUE_BATCH,INP_DIM[1]))
        for i in range(0,QUEUE_BATCH):
            data = INPUT_DATA.next()
            fts[i] = data[1]/255
            tgts[i] = data[2]
        sess.run(enqueue_op_sp, feed_dict={input_feats_sp: fts, input_labels_sp: tgts})

def enqueue_non_speech(sess):
    global FIRST_PASS_NS
        
    while True:
        if FIRST_PASS_NS==1:
            FIRST_PASS_NS = 0
            INPUT_DATA = rms(TRAIN_SCP_FILE_NON_SPEECH)
            _,READ_DATA = INPUT_DATA.next()

        fts = np.zeros((QUEUE_BATCH,INP_DIM[0], INP_DIM[1]))
        tgts =  np.zeros((QUEUE_BATCH, INP_DIM[1]))
        start = 0 
        end = 0
        count = 0
        
        while count<QUEUE_BATCH:
            if READ_DATA.shape[1]>=INP_DIM[1]:
                fts[count] = READ_DATA[:,:INP_DIM[1]]/255
                READ_DATA = READ_DATA[:,INP_DIM[1]:]
                count+=1
            else:
                _,READ_DATA = INPUT_DATA.next()
        sess.run(enqueue_op_ns, feed_dict={input_feats_ns: fts, input_labels_ns: tgts})
        
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
        _,_ = sess.run([data_batch_non_speech, target_batch_non_speech])
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
model = modify_autoenc_model(init_model)
#model = generate_model_simple_128x32()
print("Generating Test Set")
X_test, y_test = generate_test_set(TEST_SET_SIZE)
X_test = np.reshape(X_test,(TEST_SET_SIZE,INP_DIM[0],INP_DIM[1], 1))
y_test = np.reshape(y_test,(TEST_SET_SIZE,INP_DIM[1]))
y_test = np.eye(2)[y_test.astype('int64')]
y_count = Counter(y_test.flatten())
print 'Counts: ', y_count.values()[-1]/float(sum(y_count.values()))
print("Test Set Generated")

while epoch_num<MAX_EPOCHS:
    epoch_num+=1
    batch_num=0                                                                 # Batch number of current batch in training
    loss_batch=0                                                                # Cumulative loss
    acc_batch=0                                                                 # Cumulative accuracy
    MIN_LOSS=10
    log_batch=log_dir+'log_batch_epoch_%d.txt'%(epoch_num)                      # Batch-wise log file
    log_epoch = open(log_dir + 'log_epoch.txt','a')                             # epoch-wise log file
    FIRST_PASS_SP = 1
    FIRST_PASS_NS = 1
    start = time.time()
    ### Run a single epoch    
    while(batch_num<NUM_STEPS_TRAIN):
        batch_num+=1
        X_batch_speech, y_batch_speech = sess.run([data_batch_speech, target_batch_speech])                 # get an input batch 
        X_batch_non_speech, y_batch_non_speech = sess.run([data_batch_non_speech, target_batch_non_speech])                 # get an input batch 
        
        X_batch = np.concatenate((X_batch_speech, X_batch_non_speech), axis=0)
        y_batch = np.concatenate((y_batch_speech, y_batch_non_speech), axis=0)
        
        rand_keys = nprand.shuffle(range(BATCH_SIZE))
        X_batch_shuf = X_batch[rand_keys]
        y_batch_shuf = y_batch[rand_keys]
     
        X_batch = np.reshape(X_batch_shuf,(BATCH_SIZE, INP_DIM[0], INP_DIM[1], 1))
        y_batch = np.reshape(y_batch_shuf,(BATCH_SIZE, INP_DIM[1]))
        y_batch = np.array([np.eye(2)[tempvar] for tempvar in y_batch])
        y_count = Counter(y_batch.flatten())
        print 'Counts: ', y_count.values()[-1]/float(sum(y_count.values()))
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
                print(metrics_test)
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
