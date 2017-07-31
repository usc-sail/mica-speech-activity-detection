###
###     Given a wav_file and an existing labels file(assumed to be labelled until the ...
###     ... last .srt segment), appends zeros to the labels file, and writes to 
###     write_lab_path, by getting length of wav_file.

###
###     frame_len_ms = Frame length in milliseconds.
###     frame_shift_ms = Frame shift in milliseconds.
###

from __future__ import division
import subprocess
import os
import re

wav_files_path='/proj/rajat/movie_data/wavs/'
labels_path='/proj/rajat/kaldi/Adapt_RATS_VAD/adapt_feats/test/decoded_newl/labels/'
write_lab_path='/proj/rajat/kaldi/Adapt_RATS_VAD/adapt_feats/test/decoded_newl/zal/'
frame_len_ms = 25
frame_shift_ms = 10


def getLength(filename):
    result=subprocess.Popen(["ffprobe",filename], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return [x for x in result.stdout.readlines() if "Duration" in x]
count=0

for movie in os.listdir(labels_path):
    count+=1
    labels=open(labels_path+movie,'r').read().split()
    lab=[]
    for item in labels:
        lab.append(int(item))
    t=getLength(wav_files_path+movie+'.wav')
    time=re.findall('\d+:\d+:\d+.\d+',t[0])
    ts = time[0]
    time_in_sec=int(ts[0:2])*3600+int(ts[3:5])*60+int(ts[6:8])+int(ts[9:11])/100
    num_frames = math.floor(((time_in_sec*1000)-(frame_len_ms-frame_shift_ms))/frame_shift_ms)
   
    if(len(lab)>num_frames):
        print(movie[:-4])       ## Print movie name, if number of frames in label>length of movie,
        continue                ## i.e, error in .srt file.
    
    #print(str(count)+'.'+movie)  
    
    while(len(lab)<num_frames):
        lab.append(0)

    fw=open(write_lab_path+movie,'w')

    for item in lab:
        fw.write(str(item)+' ')
