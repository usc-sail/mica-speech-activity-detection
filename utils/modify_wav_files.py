from __future__ import division
import subprocess
import os

##
## Get duration of an audio/video file
##
def getDuration( av_file ):
    duration=subprocess.check_output("ffprobe -i %s -show_entries format=duration -v quiet -of csv=\"p=0\"" %av_file,shell=True)
    return float(duration[:-2])
###
### Scripts have been written to write whole directories,
### They can be modified to write only certain files, by
### passing a list of paths, and removing os.listdir in the
### for loop.
###

##
##     Extract .wav from .mkv/.mp4 files.
##     Input must be a list of strings(complete path to movie).
##

def extract_wavs_from_movie( movie_list, wav_dir ):
    for movie in movie_list:
        movieName = movie[movie.rfind('/')+1:movie.find('.')]
        os.system("ffmpeg -n -i %s -vn %s.wav" %(movie, wav_dir+movieName))




##
##     Given a .ctm file and a wav file, cuts the wav file according to 
##     segments in the .ctm file, and writes to write_dir
##

def cut_wavs_ctm_segments( ctm_dir, wav_dir, write_dir ):
    for files in os.listdir(wav_dir):
       movie = files[:-4]
       write_path = write_dir+movie+'/'
       os.system('mkdir '+write_path)
       ctm_data=[x.strip().split() for x in open(ctm_dir+movie+'.ctm').readlines()][3:]
       times=[[float(x[1]),float(x[2])] for x in ctm_data]
       count=1
       print(movie)
       for utt in times:
           num='{0:04}'.format(count)
           str_com='sox '+wav_dir+files+' '+write_path+movie+'_seg'+str(num)+'.wav trim '+str(utt[0])+' ='+str(utt[1])
           subprocess.call(str_com,shell=True)
           count+=1

##
##     Given a .wav file and segment length, cuts the wav file into equal  
##     segments, and writes to write_dir
##     segment_length is in seconds.
##

def cut_wavs_equal_segments( movie_list, wav_dir, write_dir, segment_length):
    for files in movie_list:
       write_path = write_dir + files + '/'
       os.system('mkdir -p '+write_path)
       movie_time = getDuration(wav_dir+files+'.wav')
       count = 1
       for time in range(0, int(movie_time), segment_length):
           num='{0:04}'.format(count)
           str_com='sox '+wav_dir+files+'.wav'+' '+write_path+files+'-seg_'+str(num)+'.wav trim '+str(time)+' '+str(segment_length)
           subprocess.call(str_com,shell=True)
           count+=1

##
##     Script to downsample wav files to 'down_rate' kHz
##     (default 8Khz since RAT_VAD model trained for audio files of 8kHz)
##

def downsample_wavs( wav_dir, write_dir, down_rate ):
    for fwav in os.listdir(wav_dir):
        sox_command = "sox %s -c 1 -r %f %s" %((wav_dir+fwav), down_rate, (write_dir+fwav))
        os.system(sox_command)
        #print(fwav[:-4])


if __name__ == '__main__':
    movie_paths = [x.rstrip() for x in open('movie_paths.txt')]
    ctm_dir = '/proj/rajat/movie_data/ctm_files/'
    wav_dir = '/proj/rajat/movie_data/wavs/'
    wav_ds_dir = '/proj/rajat/movie_data/wavs_ds/'
    segments_dir = '/proj/rajat/movie_data/wav_segments/'
    
    extract_wavs_from_movie(movie_paths, wav_dir)
    downsample_wavs(wav_dir, wav_ds_dir, 8000)
    cut_wavs_ctm_segments(ctm_dir, wav_dir, segments_dir)

    wav_dir='/proj/rajat/keras_model/wavs_16k/'
    movie_list = '/proj/rajat/keras_model/train/movie_list'
    write_dir='/proj/rajta/keras_model/wavs_16k_segments/'
    mov = [x.rstrip() for x in open(movie_list,'r').readlines()]
    seg_len_frm = 256
    cut_wavs_equal_segments(mov, wav_dir, write_dir, seg_len_frm)
