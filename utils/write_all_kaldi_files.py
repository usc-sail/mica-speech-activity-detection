###
###     Script to create the different kaldi input files required
###     given a list of movies(tst_list)
###     Files written - text, segments, wav.scp, utt2spk, spk2utt
### 

## Formats for different files :
## text     - <utterance-id> <utterance-transcription>
## wav.scp  - <recording-id> <extended-filename>
## segments - <utterance-id> <recording-id> <segment-begin> <segment-end>
## 
## utt2spk, spk2utt - <utterance-id> <speaker-id>
##
## For our application of VAD, recording, utterance and speaker
## id's are all the same for each file (as of NOW! not very sure)

import os
import subprocess
import sys

proj_dir = '/proj/rajat/keras_model/'
tst_list = sys.argv[1]
tst_mov = [x.rstrip() for x in open(tst_list,'r').readlines()]
wav_path = '/proj/rajat/movie_data/wavs/'
ctm_files = '/proj/rajat/movie_data/ctm_files/'
write_dir = proj_dir+'train/'
scp_file = open(write_dir+"wav.scp","w")
utt2spk_file = open(write_dir+"utt2spk","w")
text_file=open(write_dir+'text','w')
seg_file = open(write_dir+'segments','w')
for mov in tst_mov:
    fctm=[x.strip().split() for x in open(ctm_files+mov+'.ctm','r').readlines()][3:]
    labels=[int(x[3]) for x in fctm]
    time=[[x[1],x[2]] for x in fctm]
    utt=[x[4:] for x in fctm]
    
    count=0
    for i in range(len(fctm)):
        if(labels[i]==1):
            count+=1
            utt_id = mov+'-utt_'+str("{:04}".format(count))
            rec_id = mov

            utt2spk_file.write(utt_id+' '+utt_id+'\n')
            seg_file.write(utt_id+' '+rec_id+' '+time[i][0]+' '+time[i][1]+'\n')
            text_file.write(utt_id)
            for word in utt[i]:
                text_file.write(' '+word)
            text_file.write('\n')
            

    scp_file.write(mov+ ' ' + wav_path+mov+'.wav' + '\n')

text_file.close()
seg_file.close()
scp_file.close()
utt2spk_file.close()
subprocess.call('cp '+write_dir+'utt2spk '+write_dir+ 'spk2utt', shell=True)    ### spk2utt file is same as utt2spk
