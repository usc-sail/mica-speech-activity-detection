#!/bin/bash
. ./path.sh

##
##
##
##  Generate vad posteriors based on mfcc-dnn model trained on keras. 
##  Input must be a text file containing full paths to either mp4/mkv files or wav 
##  files, and optionally the path to the directory where all the output will be 
##  stored (default=$proj_dir/expt)
##  Output will be a ctm file for each movie, each line of which will contain the 
##  start and end times for the speech segment.
##  Segment-wise labels can be found at expt_dir/VAD.labels
##  E.g., ./generate_vad_labels.sh movie_path_list.txt expt_dir
##
##  Variables to be aware of:
##      nj            : number of jobs to be run in paralled (for feature-extraction)
##      expt_dir      : directory where all the output-files will be created
##      model_file    : the mfcc_model to be used for generating labels
##      ctmpath       : the path where ctm-files for each movie are written, indicating
##                      the start and end times for each speech segment on each line
##      writeVAD_file : the text file to which segment-wise labels are written to
##      labelpath     : the text file to which movie-wise labels are written to
##      med_filt      : length of median filter window to be used for median-filtering 
##                      the labels, if specified 0, posteriors will be printed to file 
##                      instead of labels
##
## 

proj_dir=/proj/audio_pipeline/scripts/standalone/rajat

movListPath=${1}
if [ $# -gt 1 ]; then
    expt_dir=$proj_dir/${2}
else
    expt_dir=$proj_dir/expt
fi

nj=10
cmd="run.pl"
wav_dir=$expt_dir/wavs
feats_dir=$expt_dir/feats
feats_log=$feats_dir/log
scpfile=$feats_dir/wav.scp
ctmpath=$expt_dir/ctm
labelpath=$expt_dir/VAD_movie.labels
py_scripts_dir=$proj_dir/python_scripts
writeVAD_file=$expt_dir/VAD.labels
model_file=/proj/rajat/keras_model/gentle_aligned_data/exp_data/exp1.1/final_nnet_30
movie_list=$expt_dir/movie.list
med_filt=11
movtime_int=0

if [ -f path.sh ]; then . ./path.sh; fi
rm -r $expt_dir
mkdir -p $wav_dir $feats_log

### Create .wavs if movie_file given
echo " >>>> CREATING WAV FILES <<<< "

for mov_file in `cat ${movListPath}`
do
    base=`basename $mov_file`
    movieName=`echo $base | awk -F '.' '{ print $1 }'`
    if [[ ${mov_file} != *"wav" ]]; then
        ffmpeg  -n -i ${mov_file} -vn tmp.wav        ## Convert .mp4/.mkv to .wav audio format
        sox tmp.wav -r 8k -c 1 ${wav_dir}/${movieName}.wav   ## Downsample .wav to 8kHz
        rm tmp.wav
    else
        sox $mov_file -r 8k -c 1 ${wav_dir}/${movieName}.wav
    fi
        
        movie_time=`soxi -D ${wav_dir}/${movieName}.wav`
        movtime_int=`echo $movie_time | awk -F '.' '{ print $1 }'`
        printf "${movieName} ${movie_time} \n" >> $expt_dir/movstats.txt
       
        seg_id=0 
        mkdir -p $wav_dir/$movieName
    ### Split into wavs of one second each
        for (( ind=0; ind<$movtime_int; ind++ ))
        do
            ((seg_id++))
            segnum=`printf "%04d" $seg_id`
            sox ${wav_dir}/${movieName}.wav $wav_dir/$movieName/${movieName}_seg-${segnum}.wav trim $ind 1.015 
        done
        rm ${wav_dir}/${movieName}.wav

done

### Create .scp file for wavs
find $wav_dir -type f -name '*.wav' | while read r
do 
    echo `basename $r .wav`" "$r 
done > $scpfile

sort $scpfile -o $scpfile

### Extract mfcc-features for wav
echo " >>>> EXTRACTING MFCC FEATURES <<<< "
steps/make_mfcc.sh --nj $nj --cmd $cmd $feats_dir $feats_log/mfcc_logs $feats_dir/mfcc_data
compute-cmvn-stats scp:$feats_dir/feats.scp ark,scp,t:$feats_dir/cmvn_stats.ark,$feats_dir/cmvn_stats.scp
apply-cmvn --norm-vars=true scp:$feats_dir/cmvn_stats.scp scp,p:$feats_dir/feats.scp ark,scp,t:$feats_dir/norm_feats.ark,$feats_dir/norm_feats.scp
steps/splice_feats_mod.sh --nj $nj --cmd $cmd $feats_dir $feats_log/splice_logs $feats_dir/mfcc_data


### Generate VAD Labels
mkdir -p $ctmpath
echo " >>>> GENERATING VAD LABELS <<<< "
python $py_scripts_dir/generate_vad_labels.py $feats_dir/spliced_feats.scp $model_file $writeVAD_file

cat $movListPath | cut -f5 -d\/ | cut -f1 -d\. > $movie_list

python $py_scripts_dir/combine_movie_seg_labels.py $movie_list $writeVAD_file $ctmpath $labelpath $med_filt
