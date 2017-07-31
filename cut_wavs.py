###
###     Given a .ctm file and a wav file, cuts the wav file according to 
###     segments in the .ctm file, and writes to write_dir
###

import os
import subprocess
ctm_dir = '/proj/rajat/movie_data/ctm_files/'
wav_dir = '/proj/rajat/movie_data/wavs/'
write_dir = '/proj/rajat/movie_data/wav_segments/'

for files in os.listdir(wav_dir):
   movie = files[:-4]
   write_path = write_dir+movie+'/'
   subprocess.call('mkdir '+write_path,shell=True)
   ctm_data=[x.strip().split() for x in open(ctm_dir+movie+'.ctm').readlines()][3:]
   times=[[float(x[1]),float(x[2])] for x in ctm_data]
   count=1
   print(movie)
   for utt in times:
       num='{0:04}'.format(count)
       str_com='sox '+wav_dir+files+' '+write_path+movie+'_seg'+str(num)+'.wav trim '+str(utt[0])+' ='+str(utt[1])
       subprocess.call(str_com,shell=True)
       count+=1
    
