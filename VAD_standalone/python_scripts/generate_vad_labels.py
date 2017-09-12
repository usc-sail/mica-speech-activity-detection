###
###     Python Script to generate VAD labels given spliced-
###     MFCC features as input
###
###     INPUTS:
###     scp_file : Input feature file
###     model_file : .h5 model file trained on keras
###     write_path : directory in which to write VAD labels
###

from __future__ import division
import os
import sys
import numpy as np
from keras.models import load_model
from scipy import signal as sig
from read_scp import read_mat_scp as rms

scp_file, model_file, write_path = sys.argv[1:]

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
model = load_model(model_file)
GLOB_SCP = '/proj/rajat/keras_model/gentle_aligned_data/feats/GLOB_MEAN_STD_MFCC.scp'

GLOB = rms(GLOB_SCP)
gen = rms(scp_file)
_, MEAN = GLOB.next()
_, STD = GLOB.next()


fw = open(write_path,'w')
for key, mat in gen:
    mat = (mat-MEAN)/STD
    pred = model.predict(mat, batch_size=50, verbose=0)
    labels = [round(x[1]) for x in pred]
    fw.write(key)
    for frame in labels:
        fw.write(" %d" %frame)
    fw.write('\n')

fw.close()


