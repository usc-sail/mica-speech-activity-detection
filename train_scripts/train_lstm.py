from __future__ import division
from numpy import random as nprand
import sys
import os
sys.path.insert(0,'/proj/rajat/keras_model/gentle_aligned_data/scripts/')
from read_scp import read_mat_scp as rms
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, BatchNormalization, Activation, Dropout, Bidirectional, LSTM
from keras.backend.tensorflow_backend import set_session
import numpy as np
import tensorflow as tf
import threading
import time
import subprocess

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#### PATHS DEFINITION
proj_dir = '/proj/rajat/keras_model/gentle_aligned_data/'
TRAIN_SCP_FILE_SP = proj_dir + 'feats/train/speech/shuf_logmel_feats.scp'
TRAIN_SCP_FILE_NS = proj_dir + 'feats/train/non_speech/shuf_logmel_feats.scp'
TEST_SCP_FILE_SP = proj_dir + 'feats/test/speech/shuf_logmel_feats.scp'
TEST_SCP_FILE_NS = proj_dir + 'feats/test/non_speech/shuf_logmel_feats.scp'
#GLOB_MEAN_STD_SCP = proj_dir+ 'feats/test/glob_std.scp'
#BAD_FEATS_LIST_FILE = proj_dir + 'feats/test/list_bad_feats'

exp_dir = proj_dir + 'exp_data/log_mel/exp2.1.1.0/'
log_dir = exp_dir + 'logs/'
log_batch = log_dir + 'log_batch_'
log_epoch = log_dir + 'log_epoch.txt'
save_model_path = exp_dir + 'final_nnet_'
os.system('mkdir -p %s' %(log_dir))
os.system('rm %s*' %(log_dir))

## Global Mean and standard deviation of input features, for standardization
#gen_std = rms(GLOB_MEAN_STD_SCP)
#_, MEAN = gen_std.next()
#_, STD = gen_std.next()
#bad_feats_seg = [x.rstrip() for x in open(BAD_FEATS_LIST_FILE).readlines()]             ## List of segments which are giving erroneous values for mean, and variance.
## Queueing-related Variables
NUM_THREADS = 16                 ## Number of threads to be used
QUEUE_BATCH = 100               ## How many samples we put in a queue at a single enqueuing process
QUEUE_CAPACITY = 1000           ## Capacity of queue
MIN_AFTER_DEQUEUE = 200         ## How big a buffer we will randomly sample from (bigger=>better shuffling but slow start-up and more memory used)
QUEUE_BATCH_CAP = QUEUE_CAPACITY
QUEUE_BATCH_MIN = MIN_AFTER_DEQUEUE

class FullyConnected(object):
    def __init__(self, inp_dim=(16,23), learning_rate=1e-4, batch_size=100, 
                    num_steps=100000, optimizer='adam', loss='binary_crossentropy',
                        stopping_criterion=1e-4):
        self.inp_dim = inp_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_steps
        self.optimizer = optimizer
        self.loss = loss
        self.loss_history = 0
        self.avg_loss = 0
        self.avg_acc = 0
        self.stopping_criterion=1e-4

        self.define_model()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def define_model(self):
        self.model = Sequential()
        layers = [ #BatchNormalization(input_shape=[self.inp_dim,]),
                   #Reshape((16,23)),
                   Bidirectional(LSTM(50), input_shape=(16,23)),
                   #Activation('relu'),
                   #BatchNormalization(),
                   #Dropout(0.4),
                   #Dense(256),
                   #Activation('relu'),
                   #BatchNormalization(),
                   #Dropout(0.2),
                   Dense(128),
                   Activation('relu'),
                   #BatchNormalization(),
                   #Dropout(0.2),
                   Dense(64),
                   Activation('relu'),
                   Dense(2),
                   Activation('softmax')
                   ]
        for layer in layers:
            self.model.add(layer)
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        print("Model Summary")
        self.model.summary()
   
    def run_single_step(self, X_batch, y_batch):
        metrics = self.model.fit(X_batch, y_batch, batch_size=self.batch_size, epochs=1, verbose=2)
        return metrics
    ### Write training loss and accuracy to file

    def report_training_stats(self, metrics, batch_num, epoch, max_batch):
        if batch_num == 0:
            with open(log_epoch,'a') as f:
                f.write("EPOCH #%d TRAINING - AVG LOSS : %0.4f, AVG ACC : %0.2f\n\t\tTIME TAKEN" \
                            "- %f seconds\n" %(epoch, self.avg_loss/max_batch,
                                self.avg_acc/max_batch, time.time()-self.start_time))
            self.start_time = time.time()
            self.loss_history = self.avg_loss
            self.avg_loss = 0
            self.avg_acc = 0

        acc = metrics.history["acc"][0]
        loss = metrics.history["loss"][0]
        self.avg_loss += loss
        self.avg_acc += acc 
        # Write avg_stats to file only every 100 batches
        if batch_num % 100 == 0:
            log_file = log_batch + str(epoch) + '.txt'
            with open(log_file,'a') as f:
                f.write("AVG LOSS : %0.4f, ACC : %0.2f after training %d batches\n" 
                    %((self.avg_loss/batch_num), (self.avg_acc/batch_num), batch_num))
    
    ### Write validation metrics to file
    def report_validation_stats(self, metrics, epoch):
        log_file = log_batch + str(epoch) + '.txt'
        with open(log_file,'a') as f:
            f.write("TESTING - LOSS : %0.4f, ACC : %0.2f\n" %(metrics[0], metrics[1]))

    ## Save model every epoch
    def save_model(self, epoch_num):
        full_path = save_model_path + str(epoch_num) + '.h5'
        self.model.save(full_path)                       

    ## Stopping criterion for training
    def stop_training(self, epoch_num):
        if(epoch_num<=15):                       # Run minimum 5 epochs
            return False
        elif((self.avg_loss-self.loss_history) < self.stopping_criterion):
            return True

class DataGeneration(object):
    ### Initialize all necessary tensorflow placeholders, 
    ### operations and threads necessary for 
    ### processing of input queues. 
    def __init__(self, sess, queue_batch=100, batch_size=50, inp_dim=(16,23)):
        self.inp_dim = np.prod(inp_dim)
        self.queue_batch = queue_batch
        self.batch_size = batch_size
        self.ts_dim = inp_dim[0]
        self.feat_dim = inp_dim[1]
        self.FIRST_PASS_SPC = 1
        self.FIRST_PASS_NS = 1
        self.build_tf_graph()       ## Define tf ops
        self.sess = sess
#        self.sess.run(tf.global_variables_initializer())
        
        self.start_threads()        ## Start threads for parallel processing of input pipelines
    
    def start_threads(self):
        self.coord = tf.train.Coordinator()
        enqueue_thread_speech = threading.Thread(target=self.enqueue_speech, args=[self.sess])
        enqueue_thread_speech.daemon = True
        enqueue_thread_speech.start()
        enqueue_thread_non_speech = threading.Thread(target=self.enqueue_non_speech, args=[self.sess])
        enqueue_thread_non_speech.daemon = True
        enqueue_thread_non_speech.start()
        self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

    def build_tf_graph(self):
        self.input_feats_speech = tf.placeholder(tf.float32, shape = [self.queue_batch, self.inp_dim])
        self.input_labels_speech = tf.placeholder(tf.int64, shape = [self.queue_batch,2])
        self.input_feats_non_speech = tf.placeholder(tf.float32, shape = [self.queue_batch, self.inp_dim])
        self.input_labels_non_speech = tf.placeholder(tf.int64, shape = [self.queue_batch,2])

        self.queue_speech = tf.FIFOQueue(capacity = QUEUE_CAPACITY, dtypes=[tf.float32, tf.int64], shapes=[[self.inp_dim],[2]])
        self.enqueue_op_speech = self.queue_speech.enqueue_many([self.input_feats_speech,self.input_labels_speech])
        self.dequeue_op_speech = self.queue_speech.dequeue_many(self.queue_batch)

        [self.data_batch_speech, self.target_batch_speech] = tf.train.shuffle_batch(self.dequeue_op_speech, batch_size=int(self.batch_size/2), 
                                    capacity=QUEUE_BATCH_CAP, min_after_dequeue=QUEUE_BATCH_MIN, enqueue_many=True)

        self.queue_non_speech = tf.FIFOQueue(capacity = QUEUE_CAPACITY, dtypes=[tf.float32, tf.int64], shapes=[[self.inp_dim], [2]])
        self.enqueue_op_non_speech = self.queue_non_speech.enqueue_many([self.input_feats_non_speech, self.input_labels_non_speech])
        self.dequeue_op_non_speech = self.queue_non_speech.dequeue_many(self.queue_batch)

        [self.data_batch_non_speech, self.target_batch_non_speech] = tf.train.shuffle_batch(self.dequeue_op_non_speech, 
                batch_size=int(self.batch_size/2), capacity=QUEUE_BATCH_CAP, min_after_dequeue=QUEUE_BATCH_MIN, enqueue_many=True)
       
    ### Put self.queue_batch number of speech samples into the queue.
    def enqueue_speech(self, sess):
        global FIRST_PASS_SPC
        while True:
            fts = np.zeros((self.queue_batch,self.inp_dim))
            start = 0
            end = 0
            if self.FIRST_PASS_SPC==1:
                self.FIRST_PASS_SPC=0
                self.data_generator_spc = rms(TRAIN_SCP_FILE_SP)
                _, self.data_spc = self.data_generator_spc.next()
    #            self.data = (self.data-MEAN)/STD

            while(end<self.queue_batch):
                end = start + len(self.data_spc)
                if(end<self.queue_batch):
                    fts[start:end] = self.data_spc[:,:self.inp_dim]
                    key, self.data_spc = self.data_generator_spc.next()
                    start = end
                else:
                    end = self.queue_batch
                    fts[start:end] = self.data_spc[:end-start,:self.inp_dim]
                    self.data_spc = self.data_spc[end-start:]
            #print("Enqueueing Speech")
            lbl = np.transpose(np.vstack((np.zeros(self.queue_batch),np.ones(self.queue_batch))))
            
            sess.run(self.enqueue_op_speech, feed_dict={self.input_feats_speech: fts, self.input_labels_speech: lbl})

    ### Put self.queue_batch number of non-speech samples into the queue.
    def enqueue_non_speech(self, sess):
        while True:
            fts = np.zeros((self.queue_batch,self.inp_dim))
            start = 0
            end = 0
            if self.FIRST_PASS_NS == 1:
                self.FIRST_PASS_NS = 0
                self.data_generator_ns = rms(TRAIN_SCP_FILE_NS)
                _, self.data_ns = self.data_generator_ns.next()
    #            self.data_ns = (self.data_ns-MEAN)/STD

            while(end<self.queue_batch):
                end = start + len(self.data_ns)
                if(end<self.queue_batch):
                    fts[start:end] = self.data_ns[:,:self.inp_dim]
                    key, self.data_ns = self.data_generator_ns.next()
                    start = end
                else:
                    end = self.queue_batch
                    fts[start:end] = self.data_ns[:end-start,:self.inp_dim]
                    self.data_ns = self.data_ns[end-start:]
            #print("Enqueueing non-speech")
            lbl = np.transpose(np.vstack((np.ones(self.queue_batch),np.zeros(self.queue_batch))))

            sess.run(self.enqueue_op_non_speech, feed_dict={self.input_feats_non_speech: fts, self.input_labels_non_speech: lbl})

    ### Generate a training input batch by
    ### generating both speech and non-speech
    ### samples.
    def generate_training_batch(self):
        X_batch_speech, y_batch_speech = self.sess.run([self.data_batch_speech, self.target_batch_speech])
        X_batch_non_speech, y_batch_non_speech = self.sess.run([self.data_batch_non_speech, self.target_batch_non_speech])
        X_batch = np.concatenate((X_batch_speech,X_batch_non_speech), axis=0)
        y_batch = np.concatenate((y_batch_speech,y_batch_non_speech), axis=0)
        
        rand_keys = np.random.permutation(range(self.batch_size))
        X_batch_shuf = X_batch[rand_keys]
        y_batch_shuf = y_batch[rand_keys]

        X_batch = np.reshape(X_batch_shuf,(self.batch_size, self.ts_dim, self.feat_dim))
        y_batch = np.reshape(y_batch_shuf,(self.batch_size,2))
        return X_batch, y_batch 

    def empty_input_queues(self):
        size = self.sess.run(self.queue_speech.size())
        self.sess.run(self.queue_speech.dequeue_many(size))
        size = self.sess.run(self.queue_non_speech.size())
        self.sess.run(self.queue_non_speech.dequeue_many(size))

    def close_threads_and_session(self):
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.run(self.queue_speech.close(cancel_pending_enqueues=True))
        self.sess.run(self.queue_non_speech.close(cancel_pending_enqueues=True))
        self.sess.close()

### Generate test set to be tested during training
### given test set size, will generate equal and random
### examples of each class
def generate_validation_set(test_set_size, inp_dim):
    gen_spc = rms(TEST_SCP_FILE_SP)
    gen_nsp = rms(TEST_SCP_FILE_NS)
    X_test = np.zeros((test_set_size, np.prod(inp_dim)))
    y_test = np.zeros((test_set_size,2))
    count = 0
    speech_flag=1
    print("...Generating Test Set")
    while True:
        if speech_flag==1:
            _, feats = gen_spc.next()
        else:
            _, feats = gen_nsp.next()
        for i in range(0,len(feats),5):
            if(count < test_set_size/2):
                X_test[count] = feats[i,:np.prod(inp_dim)]#(feats[i]-MEAN)/STD
                y_test[count] = (0, 1)
            elif(count < test_set_size):
                speech_flag=0
                X_test[count] = feats[i,:np.prod(inp_dim)]#(feats[i]-MEAN)/STD
                y_test[count] = (1, 0)
            else:
                rand_keys = nprand.permutation(range(test_set_size))
                X_test_shuf = X_test[rand_keys]
                y_test_shuf = y_test[rand_keys]
                X_test_shuf = np.reshape(X_test_shuf, [test_set_size, inp_dim[0], inp_dim[1]])
                print("Test Set Generated...")
                return X_test_shuf, y_test_shuf
            count += 1
    
######
######
## START - CODE TO DEBUG INPUT QUEUE`S
def debug_input_queues(data,sess):
    for i in range(1):
        data.FIRST_PASS_SPC = 1
        data.FIRST_PASS_NS = 1
        start = time.time()
#        print(i)
        for j in range(1000):
            [X, y] = data.generate_training_batch()
 #           print(j)
        print('%f time elapsed'%(time.time()-start))     
        size = data.sess.run(queue.size())
        sess.run(queue.dequeue_many(size))
        
## END - CODE TO DEBUG INPUT QUEUES
######
######        

### START TRAINING
def train(learning_rate=1e-4, batch_size=50, num_epoch=50, num_steps=100000, test_set_size=5000, inp_dim=(16,23)):
    sess = tf.Session(config=config)
    set_session(sess)
    fcnet = FullyConnected(learning_rate=learning_rate, batch_size=batch_size,
                                num_steps=num_steps, inp_dim=inp_dim)
    data_obj = DataGeneration(queue_batch=100, batch_size=batch_size, inp_dim=inp_dim, sess=sess)
    X_test, y_test = generate_validation_set(test_set_size, inp_dim)
    fcnet.start_time = 0     
    for epoch in range(num_epoch):
        data_obj.FIRST_PASS_SPC = 1
        data_obj.FIRST_PASS_NS = 1
        #initialize per_epoch variables
        for batch in range(num_steps):
            X_batch, y_batch = data_obj.generate_training_batch()
            # Train on single batch
            train_metrics = fcnet.run_single_step(X_batch, y_batch)
            if batch % 10000 == 0:   ## Evaluate model on test set every 10000 batches
                test_metrics = fcnet.model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
                fcnet.report_validation_stats(test_metrics, epoch)

            fcnet.report_training_stats(train_metrics, batch, epoch, num_steps)

        fcnet.save_model(epoch)             ## Save model
        ### Empty input queues
        data_obj.empty_input_queues()

        if fcnet.stop_training(epoch) is True: 
            break                           ## Stop training if loss is not decreasing
    

    data_obj.close_threads_and_session()
    return fcnet.model

if __name__ == '__main__':
    global FIRST_PASS_SPC, FIRST_PASS_NS
    FIRST_PASS_SPC = FIRST_PASS_NS = 1
    model = train(learning_rate=5e-4, batch_size=50, num_epoch=50, 
            num_steps=100000, test_set_size=2000, inp_dim = (16,23))

