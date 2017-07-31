## Function to parse a subtitle(.srt) file to a .ctm file
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
##                2 - completely paranthesized segment            
## utterance    : sentence uttered as given in .srt file(lot of typos!)
##                (may be unnecessary)
## 

from __future__ import division
import re
import os

def write_ctm_files_vad( srt_path, ctm_path, buffer, threshold ):
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
        print(str(fname[index+1:-5]))
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
        x_comp = [ (x[i-1][1]+buffer, x[i][0]-buffer) for i in range(1,len(x))]
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
                    uttline = uttline+lines[countl]
                    countl+=1
                x1[countu].append(uttline)
                if uttline!='':
                    if((uttline[0]=='(' and uttline[-1]==')') or (uttline[0]=='[' and uttline[-1]==']')):         
                        x1[countu][2] = 2
                    else:
                        x1[countu][2] = 1
                countu += 1
            else:
                countl += 1
    
        data = x1+x0                            ## combine voiced and unvoiced segments
        data.sort(key = lambda x:x[0])          ## merge the segments based on start time to get final data list to be written to .ctm file
        ### Writitng into ctm files:
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
    print(count)    
        
srtpath = open("../Data_Paths/srt_files.txt","r").readlines()
destpath = '/proj/rajat/ctm_files_3class/'
buf = 0
thr = 0
write_ctm_files_vad(srtpath,destpath,buf,thr)
