from __future__ import division
import subprocess
import re
import os

# Function to get duration of a movie 
def getDuration(filename):

    result=subprocess.Popen(["ffprobe",filename], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return [x for x in result.stdout.readlines() if "Duration" in x]
count=0


# Function to parse a subtitle(.srt) file to a .ctm file
## srt_path  -  Path to folder containing .srt files
## ctm_path  -  Path to folder where .ctm files are to be created
## buffer    -  Time in seconds used as buffer before and after each voiced frame
## threshold -  Time in seconds between two voiced frames(after removing buffer),
##              needed to consider it as unvoiced frame 
##
## The format of the .ctm file used is 
## <recording-id> <begin-time> <end-time> <voiced/unvoiced label> <utterance>
## 
## recording-id : typically use the movie name as seen in the srt/wav files
## begin-time   : start time of segment as given in .srt file
## end-time     : end time of segment as given in .srt file
## labels       : 0 - unvoiced segment
##                1 - voiced segment
## utterance    : sentence uttered as given in .srt file(lot of typos!)
##                (may be unnecessary)
## 

def write_ctm_files( srt_path, ctm_path, buffer_ , threshold ):
    TOT_TIME = 0                                    # Total number of hours 
    TOT_SPC = 0                                     # Amount of Speech Data (in hours) 
    TOT_NSE = 0                                     # Amount of Noise Data (in hours)
    flog = open(ctm_path + "logfile.txt","w")       # Logfile for data log
    flog.write("%%%  FILE_NAME  PERCENT_SPEECH PERCENT_NON_SPEECH TOTAL_LENGTH %%%\n") 
    count=0
    for fname in srt_path:
        count+=1
        f = open(fname[:-1], "r")                   # Open .srt files to read from
        index = str(fname).rfind('/')
        fw = open(ctm_path + fname[index+1:-5] + ".ctm", "w")           # Output .ctm files
#        print(str(fname[index+1:-5]))
        ### Read .srt files
        content = f.read()
        start_time = re.findall("\d+:\d+:\d+,\d+ -", content)   # raw start times
        end_time = re.findall("> \d+:\d+:\d+,\d+", content)     # raw end times
        ### Parsing of Time
        start_time_seconds = []                                 # parsed start times
        end_time_seconds = []                                   # parsed end times
        duration = []                                           
        spc_time = 0                                            # amount of speech data in this file(in seconds)
        nse_time = 0                                            # amount of noise data in this file(in seconds)
        tot_time = 0                                            # total length of movie(in seconds?)

        for i in range(0, len(start_time)):
            st = start_time[i]
            et = end_time[i]
            starttimeinseconds = int(st[0:2])*3600 + int(st[3:5])*60 + int(st[6:8]) + int(st[9:12])/1000
            endtimeinseconds = int(et[2:4])*3600 + int(et[5:7])*60 + int(et[8:10]) + int(et[11:14])/1000
            durationinseconds = endtimeinseconds - starttimeinseconds
            start_time_seconds.append(starttimeinseconds)
            end_time_seconds.append(endtimeinseconds)
            duration.append(format(durationinseconds,'.3f'))
        tot_time = end_time_seconds[-1]        

        ### Handling of unvoiced segments
        x = [(start_time_seconds[i], end_time_seconds[i]) for i in range(0,len(start_time_seconds))]
        x_comp = [ (x[i-1][1]+buffer_, x[i][0]-buffer_) for i in range(1,len(x))]
        x_valid = [i for i in x_comp if(i[1]-i[0])>threshold]

        x0 = [[float(format(i[0],'0.3f')), float(format(i[1],'0.3f')), 0, "NOISE"] for i in x_valid]          # Labelled Unvoiced
        x1 = [[i[0], i[1], 1] for i in x]                                                                     # Labelled Voiced

        ### Beginning comments in .ctm files
        fw.write(';;\n')
        fw.write(';;  <recording-id> <begin-time> <end-time> <label> <utterance>   ;;\n')
        fw.write(';;\n')

        ### Parse Utterances and write .ctm files
        f.seek(0)
        lines = f.readlines()

        data=[x.strip() for x in lines]
        lines=data
        countl = 0          # Line number in .srt file
        countu = 0          # Utterance number 
        while(countl<len(lines)):
            if re.match("\d+:\d+:\d+,\d+ --> \d+:\d+:\d+,\d+", lines[countl]):
                countl += 1
                uttline = ''
                while(lines[countl]!='' and countl < len(lines)-1):      ## loop to extract multiple lines in a single utterance(if multiple are found,
                    uttline = uttline+' '+lines[countl]
                    countl+=1
                x1[countu].append(uttline)
                if uttline!='':
                    ### Most heuristic part of code, label a segment as unvoiced if the utterance is surrounded by braces( '()'' or '[]' ), or 
                    ### if the utterance contains the character '\xe2' (get's rid of the music parts)  
                    if(('(' in uttline and ')' in uttline)):   ## used to remove music, left out for now-->  or re.search("\xe2",uttline) ):
                        x1[countu][2] = 2
                else:
                    x1[countu][2] = 0
                countu += 1
            else:
                countl += 1
    
        data = x1+x0                            ## combine voiced and unvoiced segments
        data.sort(key = lambda x:x[0])          ## merge the segments based on start time to get final data list to be written to .ctm file
        ### Writitng into ctm files:
        first_line = [[0, start_time_seconds[0], 0, "NOISE"]]
        data = first_line+data
        for line in data:
            fw.write(fname[index+1:-5]+ ' ')
            for item in line:
                fw.write(str(item)+ ' ')
            fw.write('\n')
        
        ### Updating logfile parameters and writing to logfile
        x0_diff = [] 
        nse_time_temp =[]
        spc_time = [i[1]-i[0] for i in x1 if i[2]==1]
        spc_time = sum(spc_time)
        nse_time_temp = [i[1]-i[0] for i in x1 if i[2]==0]
        x0_diff = [i[1]-i[0] for i in x0]
        nse_time = sum(nse_time_temp)+sum(x0_diff)
        tot_spc_per = format(spc_time/tot_time*100,'0.2f')
        tot_nse_per = format(nse_time/tot_time*100,'0.2f')
        spc_pl_nse = spc_time+nse_time

        TOT_TIME+=tot_time
        TOT_SPC+=spc_time
        TOT_NSE+=nse_time
    
        flog.write(fname[index+1:-5] + ' ' + str(tot_spc_per) + ' ' + str(tot_nse_per)+ ' ' + str(spc_pl_nse) + '\n')
        fw.close()
        f.close()
    
    ### Accumulated Log Data
    SPC_PER_SET = format(TOT_SPC/TOT_TIME*100,'0.2f')
    NSE_PER_SET = format(TOT_NSE/TOT_TIME*100,'0.2f')
    flog.write("\nTOTAL TIME IN HOURS : " + str(format(float(TOT_TIME)/3600,'0.2f')))
    flog.write("\nTOTAL PERCENTAGE OF SPEECH : " + str(SPC_PER_SET))
    flog.write("\nTOTAL PERCENTAGE OF NON-SPEECH : " + str(NSE_PER_SET))
    flog.write("\nTOTAL DATA IN HOURS : " + str(format((TOT_SPC+TOT_NSE)/3600,'0.2f')))
    flog.close()
    #print(count)    


###
### Add last unvoiced segment (from last subtitle segment ...
### ... till end of movie), using wav_file to extract end time
### of movie, and append to existing .ctm files.
###
### This can be used to modify .ctm file directly, instead of 
### appending zeros to labels(as in append_labels_zeros.py), 
### and then extract labels from .ctm file
###

def append_last_segment( wav_files_path, ctm_path ):
    for movie in sorted(os.listdir(ctm_path)):
        if not movie.endswith('.ctm'):
            continue
     #   print(movie)
        count+=1
        t=getDuration(wav_files_path+movie[:-4]+'.wav')
        time=re.findall('\d+:\d+:\d+.\d+',t[0])
        ts = time[0]
        time_in_sec=int(ts[0:2])*3600+int(ts[3:5])*60+int(ts[6:8])+int(ts[9:11])/100
        data = open(ctm_path+movie[:-4]+'.ctm','r').readlines()
        st_time = float(data[-1].split()[2])
        
        fw = open(ctm_path+movie,'a')
        fw.write(movie[:-4]+' '+str(st_time)+' '+str(time_in_sec)+' 0 NOISE')



###
### Generate labels from .ctm file, and write
### them to label_dir, with each frame level
### label seperated by delimiter
###
### delimiter - Delimiter used to seperate frame level labels.
### frame_shift_ms - Frame shift in milliseconds.
### frame_len_ms - Frame length in milliseconds.
###

def ctm2labels( ctm_dir, label_dir, frame_shift_ms, frame_len_ms, delimiter ):
    for ctm in os.listdir(ctm_dir):
        if not ctm.endswith('.ctm'):
            continue
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


if __name__ == '__main__':
    srt_path = open("/proj/rajat/srt_wav_Paths/srt_files.txt","r").readlines()
    wav_files_path='/data/movie_wavs/animated/'
    ctm_path='/proj/rajat/animated_ctm/'
    buf = 0
    thr = 0
    label_dir = '/proj/rajat/movie_datalabels'
    ctm_dir = '/proj/rajat/movies_data/ctm_files/'
    delimiter = ' '
    frame_shift_ms = 10
    frame_len_ms = 25
    
    write_ctm_files_vad(srt_path, ctm_path, buf, thr)
    append_last_segment(wav_files_path, ctm_path)
    ctm2labels(ctm_dir, label_dir, frame_shift_ms, frame_len_ms, delimiter)
