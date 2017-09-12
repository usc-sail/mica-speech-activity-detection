###
###     Output of our VAD prediction-process are multiple 
###     consecutive segments from the same movie. Here, we 
###     combine these segments from each movie, and store 
###     them in two easily-readable formats
###
###     labels  - frame-wise labels are printed space-seperated
###         following the key(movie name), on every line.
###     ctm     - ctm-style files are created for each movie,
###         wherein the start-end times for each speech segment
###         are printed.
###


from __future__ import division
import os
import sys
import numpy as np
from scipy import signal as sig

movlist, labelfile, ctmpath, labelpath = sys.argv[1:-1]
med_filt = int(sys.argv[-1])

movies = [x.rstrip() for x in open(movlist).readlines()]
data = [x.rstrip().split() for x in open(labelfile).readlines()]

mov_dict = {k:[] for k in movies}

for x in data:
    for movie in movies:
        if(movie in x[0]):
            for y in x[1:]:
                mov_dict[movie].append(int(y))

for movie in movies:
    if movie not in mov_dict:
        continue
    labels = mov_dict[movie]
    med_filt_labels = sig.medfilt(labels, med_filt)
    labels = list(med_filt_labels)
    #print(len(labels)) 
    fw = open(labelpath,'a')
    fw.write(movie+' ')
    for frame in labels:
        fw.write("%d " %frame)
    fw.write('\n')
    fw.close()
    start_ind = labels.index(1)
    fw = open(ctmpath+'/'+movie+'.ctm','w')

    while 1:
        try:
            end_ind = labels[start_ind:].index(0)+start_ind-1
            fw.write("%0.2f\t%0.2f\n" %(start_ind/100, end_ind/100))
            start_ind = labels[end_ind+1:].index(1)+end_ind+1
        except:
            fw.close()
            break

