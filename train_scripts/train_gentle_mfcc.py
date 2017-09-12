from __future__ import division
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense
from numpy import random as nprand
from read_scp import read_mat_scp as rms
import os
import numpy as np
import tensorflow as tf
import keras.models
import threading
import random
import time
import subprocess

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
##### DEFINITION OF PATHS AND GLOBAL VARIABLES
## Path Variables
proj_dir = '/proj/rajat/keras_model/gentle_aligned_data/'
TRAIN_SCP_FILE_SP = proj_dir + 'feats/train/speech/shuf_mfcc_feats.scp'
TRAIN_SCP_FILE_NS = proj_dir + 'feats/train/non_speech/shuf_mfcc_feats.scp'
TEST_SCP_FILE_SP = proj_dir + 'feats/test/speech/shuf_mfcc_feats.scp'
TEST_SCP_FILE_NS = proj_dir + 'feats/test/non_speech/shuf_mfcc_feats.scp'
GLOB_MEAN_STD_SCP = proj_dir+ 'feats/GLOB_MEAN_STD_MFCC.scp'
BAD_FEATS_LIST_FILE = proj_dir + 'feats/list_bad_feats'

exp_dir = proj_dir + 'exp_data/exp1/'
log_dir = exp_dir + 'logs/'
os.system('mkdir -p %s' %(log_dir))
os.system('rm %s*' %(log_dir))

## Global Mean and standard deviation of input features, for standardization
gen_std = rms(GLOB_MEAN_STD_SCP)
_, MEAN = gen_std.next()
_, STD = gen_std.next()
bad_feats_seg = [x.rstrip() for x in open(BAD_FEATS_LIST_FILE).readlines()]
## Network Architecture
INP_DIM = 403
BATCH_SIZE = 50
## Queueing-related Variables
NUM_THREADS = 4                ## Number of threads to be used
QUEUE_BATCH = 100              ## How many samples we put in a queue at a single enqueuing process
QUEUE_CAPACITY = 1000         ## Capacity of queue
MIN_AFTER_DEQUEUE = 200      ## How big a buffer we will randomly sample from (bigger=>better shuffling but slow start-up and more memory used
QUEUE_BATCH_CAP = QUEUE_CAPACITY
QUEUE_BATCH_MIN = MIN_AFTER_DEQUEUE
### Start From Here

def initialize_training_variables():
    global LEARNING_RATE, DECAY, MOMENTUM, NUM_SAMPLES_TRAIN, MAX_EPOCHS, NUM_STEPS_TRAIN, TEST_SET_SIZE, MIN_LOSS, epoch_num, AVG_LOSS_EPOCH
    LEARNING_RATE = 0.001
    DECAY = 1e-6
    MOMENTUM = 0.9
    NUM_SAMPLES_TRAIN = 1500000
    MAX_EPOCHS = 50
    NUM_STEPS_TRAIN = 100000
    TEST_SET_SIZE = 10000
    MIN_LOSS = 1
    epoch_num=0
    AVG_LOSS_EPOCH = np.zeros(MAX_EPOCHS+1)

def initialize_model():
   model = Sequential()
   model.add(Dense(512, input_dim=INP_DIM, activation='relu'))
   model.add(Dense(512, activation='relu'))
   model.add(Dense(512, activation='relu'))
   model.add(Dense(2, activation='softmax'))
   sgd = optimizers.SGD(lr=LEARNING_RATE, decay=DECAY, momentum=MOMENTUM)
   model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
   return model

def generate_test_set(test_set_size):
    #subprocess.call('shuf -o %s %s'%(TEST_SCP_FILE,TEST_SCP_FILE), shell=True)
    gen_spc = rms(TEST_SCP_FILE_SP)
    gen_nsp = rms(TEST_SCP_FILE_NS)
    X_test = np.zeros((test_set_size,INP_DIM))
    y_test = np.zeros((test_set_size,2))
    count = 0
    speech_flag=1
    while True:
        if speech_flag==1:
            _, feats = gen_spc.next()
        else:
            _, feats = gen_nsp.next()
        for i in range(0,len(feats),5):
            if(count < test_set_size/2):
                X_test[count] = (feats[i]-MEAN)/STD
                y_test[count] = (0, 1)
            elif(count < test_set_size):
                speech_flag=0
                X_test[count] = (feats[i]-MEAN)/STD
                y_test[count] = (1, 0)
            else:
                rand_keys = nprand.shuffle(range(test_set_size))
                X_test_shuf = X_test[rand_keys]
                y_test_shuf = y_test[rand_keys]
                #print("Generated Test Set of size %d" %test_set_size)
                #print("Fraction of Label 1 in test_set = %0.2f" %(sum(y_test)/len(y_test)))
                return X_test, y_test
            count += 1

def save_model(avg_loss, model, epoch_num):
    global MIN_LOSS
    if(float(avg_loss)<MIN_LOSS):
        MIN_LOSS = avg_loss
        nnet_name = 'final_nnet_'+str(epoch_num)
        model.save(exp_dir+nnet_name)

def stopping_criterion(loss, epoch_num):
    if(epoch_num==1):
        return False
    elif((loss[epoch_num-1]-loss[epoch_num])<0.0005):
        return True
        
input_feats_speech = tf.placeholder(tf.float32, shape = [QUEUE_BATCH, INP_DIM])
input_labels_speech = tf.placeholder(tf.int64, shape = [QUEUE_BATCH,2])
input_feats_non_speech = tf.placeholder(tf.float32, shape = [QUEUE_BATCH, INP_DIM])
input_labels_non_speech = tf.placeholder(tf.int64, shape = [QUEUE_BATCH,2])

queue_speech = tf.FIFOQueue(capacity = QUEUE_CAPACITY, dtypes=[tf.float32, tf.int64], shapes=[[INP_DIM],[2]])
enqueue_op_speech = queue_speech.enqueue_many([input_feats_speech,input_labels_speech])
dequeue_op_speech = queue_speech.dequeue_many(QUEUE_BATCH)

[data_batch_speech, target_batch_speech] = tf.train.shuffle_batch(dequeue_op_speech, batch_size=int(BATCH_SIZE/2), capacity=QUEUE_BATCH_CAP, min_after_dequeue=QUEUE_BATCH_MIN, enqueue_many=True)

queue_non_speech = tf.FIFOQueue(capacity = QUEUE_CAPACITY, dtypes=[tf.float32, tf.int64], shapes=[[INP_DIM], [2]])
enqueue_op_non_speech = queue_non_speech.enqueue_many([input_feats_non_speech, input_labels_non_speech])
dequeue_op_non_speech = queue_non_speech.dequeue_many(QUEUE_BATCH)

[data_batch_non_speech, target_batch_non_speech] = tf.train.shuffle_batch(dequeue_op_non_speech, batch_size=int(BATCH_SIZE/2), capacity=QUEUE_BATCH_CAP, min_after_dequeue=QUEUE_BATCH_MIN, enqueue_many=True)

## Put input batch_data of length QUEUE_BATCH into the queue.
def enqueue_speech(sess):
    global FIRST_PASS_SPC
    while True:
        fts = np.zeros((QUEUE_BATCH,INP_DIM))
        start = 0
        end = 0
        if FIRST_PASS_SPC==1:
            FIRST_PASS_SPC=0
            INPUT_DATA = rms(TRAIN_SCP_FILE_SP)
            key, READ_DATA = INPUT_DATA.next()
            READ_DATA = (READ_DATA-MEAN)/STD

        while(end<QUEUE_BATCH):
            end = start + len(READ_DATA)
            if(end<QUEUE_BATCH):
                fts[start:end] = READ_DATA[:]
                key, READ_DATA = INPUT_DATA.next()
                while key in bad_feats_seg:
                    key, READ_DATA = INPUT_DATA.next()
                READ_DATA = (READ_DATA-MEAN)/STD
                start = end
            else:
                end = QUEUE_BATCH
                fts[start:end] = READ_DATA[:end-start]
                READ_DATA = READ_DATA[end-start:]

        lbl = np.transpose(np.vstack((np.zeros(QUEUE_BATCH),np.ones(QUEUE_BATCH))))
        
        sess.run(enqueue_op_speech, feed_dict={input_feats_speech: fts, input_labels_speech: lbl})

def enqueue_non_speech(sess):
    global FIRST_PASS_NS
    while True:
        fts = np.zeros((QUEUE_BATCH,INP_DIM))
        start = 0
        end = 0
        if FIRST_PASS_NS==1:
            FIRST_PASS_NS=0
            INPUT_DATA = rms(TRAIN_SCP_FILE_NS)
            _, READ_DATA = INPUT_DATA.next()
            READ_DATA = (READ_DATA-MEAN)/STD

        while(end<QUEUE_BATCH):
            end = start + len(READ_DATA)
            if(end<QUEUE_BATCH):
                fts[start:end] = READ_DATA[:]
                key, READ_DATA = INPUT_DATA.next()
                while key in bad_feats_seg:
                    key, READ_DATA = INPUT_DATA.next()
                READ_DATA = (READ_DATA-MEAN)/STD
                start = end
            else:
                end = QUEUE_BATCH
                fts[start:end] = READ_DATA[:end-start]
                READ_DATA = READ_DATA[end-start:]
        
        lbl = np.transpose(np.vstack((np.ones(QUEUE_BATCH),np.zeros(QUEUE_BATCH))))

        sess.run(enqueue_op_non_speech, feed_dict={input_feats_non_speech: fts, input_labels_non_speech: lbl})


sess = tf.Session()
sess.run(tf.global_variables_initializer())
FIRST_PASS_SPC = 1
FIRST_PASS_NS = 1
enqueue_thread_speech = threading.Thread(target=enqueue_speech, args=[sess])
enqueue_thread_speech.daemon = True
enqueue_thread_speech.start()
enqueue_thread_non_speech = threading.Thread(target=enqueue_non_speech, args=[sess])
enqueue_thread_non_speech.daemon = True
enqueue_thread_non_speech.start()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

######
######
## START - CODE TO DEBUG INPUT QUEUES

ct = 0
while ct<1:
    break
    ct+=1
    FIRST_PASS = 1
    start = time.time()
    for i in range(1):
        b_feats_speech= sess.run([data_batch_speech])
        b_feats_non_speech = sess.run([data_batch_non_speech])
        print(b_feats_speech)
        print(b_feats_non_speech)
    print('%f time elapsed'%(time.time()-start))     
    size = sess.run(queue.size())
    sess.run(queue.dequeue_many(size))
        

## END - CODE TO DEBUG INPUT QUEUES
######
######        

### START TRAINING
initialize_training_variables()
model = initialize_model()
print("Generating Test Set")
X_test, y_test = generate_test_set(TEST_SET_SIZE)
print("Test Set Generated")

while epoch_num<MAX_EPOCHS:
    epoch_num+=1
    batch_num=0
    loss_batch=0 
    acc_batch=0
    log_batch=log_dir+'log_batch_epoch_%d.txt'%(epoch_num)
    log_epoch = open(log_dir + 'log_epoch.txt','a')
    ## Initialize queue variables for each epoch
    FIRST_PASS_SPC=1
    FIRST_PASS_NS=1
    MIN_LOSS = 1
    start = time.time()
    ### Run a single epoch    
    while(batch_num<NUM_STEPS_TRAIN):
        batch_num+=1
        X_batch_speech, y_batch_speech = sess.run([data_batch_speech, target_batch_speech])
        X_batch_non_speech, y_batch_non_speech = sess.run([data_batch_non_speech, target_batch_non_speech])
        X_batch = np.concatenate((X_batch_speech,X_batch_non_speech), axis=0)
        y_batch = np.concatenate((y_batch_speech,y_batch_non_speech), axis=0)
        
        rand_keys = np.random.shuffle(range(BATCH_SIZE))
        X_batch_shuf = X_batch[rand_keys]
        y_batch_shuf = y_batch[rand_keys]

        X_batch = np.reshape(X_batch_shuf,(BATCH_SIZE,INP_DIM))
        y_batch = np.reshape(y_batch_shuf,(BATCH_SIZE,2))

        metrics = model.fit(X_batch, y_batch, batch_size=BATCH_SIZE, epochs=1, verbose=2)
        acc = float(format(metrics.history["acc"][0],'0.2f'))
        loss = float(format(metrics.history["loss"][0],'0.2f'))
        loss_batch+=loss
        acc_batch+=acc
        if(batch_num%100==0):
            #avg_loss_batch = format(loss_batch/batch_num,'0.4f')
            #avg_acc_batch = format(acc_batch/batch_num, '0.2f')
            with open(log_batch,'a') as f:
                f.write("AVG LOSS : %0.4f, ACC : %0.2f after training %d batches\n" %((loss_batch/batch_num), (acc_batch/batch_num), batch_num))
            avg_loss_batch = loss_batch/batch_num
            save_model(avg_loss_batch, model, epoch_num)
            ## Evaluate on test set
            if(batch_num%1000==0):
                metrics_test = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=1)
                with open(log_batch,'a') as f:
                    f.write("TESTING - LOSS : %0.4f, ACC : %0.2f\n" %(metrics_test[0], metrics_test[1]))
    ### End of one epoch, stop queues and write epoch log
    AVG_LOSS_EPOCH[epoch_num] = loss_batch/batch_num
    log_epoch.write("EPOCH #%d TRAINING - AVG LOSS : %0.4f, AVG ACC : %0.2f\n\t\tTIME TAKEN - %f seconds\n" %(epoch_num, loss_batch/batch_num, acc_batch/batch_num, time.time()-start))
    log_epoch.close()
    if stopping_criterion(AVG_LOSS_EPOCH, epoch_num):
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
