### Append srt_paths and wav_paths to srt_files and wav_files
### , given new data directories corresponding
### to srt_files, and their corresponding wav_files


import os
srt_path = "/data/GDI/Film2014/subtitles/srt/"
wav_path = "/data/movie_wavs/Film2014/eng/"

count = 0
srt_files = open("../Data_Paths/srt_files.txt","a")
wav_files = open("../Data_Paths/wav_files.txt","a")
for fsrt in os.listdir(srt_path):
    for fwav in os.listdir(wav_path):
        if(fwav[:-8]==fsrt[:-4] or fwav[:-4]==fsrt[:-4]):
            count+=1 
            srt_files.write(srt_path+str(fsrt)+'\n')
            wav_files.write(wav_path+str(fwav)+'\n')
            break 
print(count)
