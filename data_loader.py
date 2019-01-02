import tensorflow as tf
import numpy as np
import glob
import os
from SAD_parameters import *

class Data:
    def __init__(self, inp_shape=INPUT_SHAPE, feat_dim=FEAT_DIM, batch_size=BATCH_SIZE, val_size=NUM_VAL_SAMPLES, 
                    speech_dp_train=DATA_PATH_SP_TRAIN, non_speech_dp_train=DATA_PATH_NS_TRAIN,
                    speech_dp_val=DATA_PATH_SP_VAL, non_speech_dp_val=DATA_PATH_NS_VAL,
                    speech_dp_test=DATA_PATH_SP_TEST, non_speech_dp_test=DATA_PATH_NS_TEST):
        self.inp_shape = inp_shape
        self.feat_dim = feat_dim
        self.batch_size = batch_size
        self.val_size = val_size
        self.data_paths_sp_train = speech_dp_train
        self.data_paths_ns_train = non_speech_dp_train
        self.data_paths_sp_val = speech_dp_val
        self.data_paths_ns_val = non_speech_dp_val
        self.data_paths_sp_test = speech_dp_test
        self.data_paths_ns_test = non_speech_dp_test
        self.data = {}
        self.build_datasets()
        
    def feature_parser(self, utt, label):
        context_features = {'feature_id': tf.FixedLenFeature([], tf.string)}
        sequence_features = {'log_mels': tf.FixedLenSequenceFeature([], tf.string)}
        [utt_id, features_raw] = tf.parse_single_sequence_example(utt,
                context_features=context_features, 
                sequence_features=sequence_features)
        features = tf.reshape(tf.decode_raw(features_raw['log_mels'],tf.float32), self.feat_dim)
        return utt_id['feature_id'], features, np.eye(2)[label]
    
    def create_TFRDataset(self, path_to_tfrecords, mode, label):
        files = glob.glob(os.path.join(path_to_tfrecords, '*tfrecord'))
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(lambda x: self.feature_parser(x, 2*[label]))
        if mode == 'train':
            dataset = dataset.shuffle(self.batch_size)
            dataset = dataset.repeat()
            dataset = dataset.batch(batch_size=int(self.batch_size/4))
            dataset = dataset.prefetch(buffer_size=self.batch_size)
        elif mode == 'val':
            dataset = dataset.batch(batch_size=self.val_size)
        elif mode == 'test':
            dataset = dataset.batch(self.batch_size/2)
        dataset_iterator = dataset.make_one_shot_iterator()
        return dataset, dataset_iterator
    
    def normalize_batch(self, data):
        mean, var = tf.nn.moments(data, axes=[0,1,2])
        norm_data = (data - mean) / tf.sqrt(var + 1e-8)
        return norm_data
    
    def concatenate_and_normalize_batch(self, sp_itr, ns_itr):
        _, X_sp, y_sp = sp_itr.get_next()
        _, X_ns, y_ns = ns_itr.get_next()
        X_batch = tf.concat((X_sp, X_ns), axis=0)
        X_batch = self.normalize_batch(X_batch)
        X_batch = tf.reshape(X_batch, tf.concat(([-1], self.inp_shape), axis=0))
        y_batch = tf.concat((y_sp, y_ns), axis=0)
        y_batch = tf.reshape(tf.cast(y_batch, tf.int32), [-1, 2])
        return X_batch, y_batch
    
    def generate_test_batch(self, sp_ds, ns_ds):
        dataset = sp_ds.concatenate(ns_ds)
        iterator = dataset.make_one_shot_iterator()
        _, X_test, y_test = iterator.get_next()
        X_test = self.normalize_batch(X_test)
        X_test = tf.reshape(X_test, tf.concat(([-1], self.inp_shape), axis=0))
        y_test = tf.reshape(tf.cast(y_test, tf.int32), [-1, 2])
        return X_test, y_test
        
    def build_datasets(self):   
        _, self.data['speech_train'] = self.create_TFRDataset(self.data_paths_sp_train, 'train', 1)
        _, self.data['non_speech_train'] = self.create_TFRDataset(self.data_paths_ns_train, 'train', 0)
        _, self.data['speech_val'] = self.create_TFRDataset(self.data_paths_sp_val, 'val', 1)
        _, self.data['non_speech_val'] = self.create_TFRDataset(self.data_paths_ns_val, 'val', 0)
        self.data['speech_test'], _ = self.create_TFRDataset(self.data_paths_sp_test, 'test', 1)
        self.data['non_speech_test'], _ = self.create_TFRDataset(self.data_paths_ns_test, 'test', 0)
        [self.X_batch, self.y_batch] = self.concatenate_and_normalize_batch(self.data['speech_train'], self.data['non_speech_train'])
        [self.X_val, self.y_val] = self.concatenate_and_normalize_batch(self.data['speech_val'], self.data['non_speech_val'])       
        [self.X_test, self.y_test] = self.generate_test_batch(self.data['speech_test'], self.data['non_speech_test'])
