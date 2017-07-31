###
### Generate labels from .ctm file, and write
### them to label_dir, with each frame level
### label seperated by delimiter
###
### delimiter - Delimiter used to seperate frame level labels.
### frame_shift_ms - Frame shift in milliseconds.
### frame_len_ms - Frame length in milliseconds.
###

from __future__ import division
import os
data_dir = '/proj/rajat/movie_data/'
ctm_dir = data_dir+'ctm_files/'
label_dir = data_dir+'labels/'
delimiter = ' '
frame_shift_ms = 10
frame_len_ms = 25

for ctm in os.listdir(ctm_dir):
    mov=ctm[:-4]
    print(mov)
    ctm_data = [x.strip().split() for x in open(ctm_dir+ctm,'r').readlines()][3:]
    times = [[float(x[1]),float(x[2])] for x in ctm_data]
    labl = [int(x[3]) for x in ctm_data]

    time_sec = 0
    utt = 0
    labels=[]

    while(time_sec<times[0][0]):
        labels.append(0)
        time_sec += frame_shift_ms/1000

    thr=times[0][1]
    while(time_sec<(times[-1][-1]-frame_len_ms/1000)):
        if(time_sec>thr):
            utt+=1
            thr=times[utt][1]
        labels.append(labl[utt])
        time_sec += frame_shift_ms/1000

    fw=open(label_dir+mov,'w')
    for lab in labels:
        fw.write(str(lab)+delimiter)
