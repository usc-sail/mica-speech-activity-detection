###
### Add last unvoiced segment (from last subtitle segment ...
### ... till end of movie), using wav_file to extract end time
### of movie, and append to existing .ctm files.
###
### This can be used to modify .ctm file directly, instead of 
### appending zeros to labels(as in append_labels_zeros.py), 
### and then extract labels from .ctm file
###

from __future__ import division
import subprocess
import os
import re

wav_files_path='/proj/rajat/movie_data/wavs/'
ctm_path='/proj/rajat/movie_data/temp_ctm/'

def getLength(filename):
    result=subprocess.Popen(["ffprobe",filename], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return [x for x in result.stdout.readlines() if "Duration" in x]
count=0

for movie in os.listdir(ctm_path):
    count+=1
    t=getLength(wav_files_path+movie[:-4]+'.wav')
    time=re.findall('\d+:\d+:\d+.\d+',t[0])
    ts = time[0]
    time_in_sec=int(ts[0:2])*3600+int(ts[3:5])*60+int(ts[6:8])+int(ts[9:11])/100
    data = open(ctm_path+movie,'r').readlines()
    st_time = float(data[-1].split()[2])
    
    fw = open(ctm_path+movie,'a')
    fw.write(movie[:-4]+' '+str(st_time)+' '+str(time_in_sec)+' 0 NOISE')
