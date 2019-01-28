#!/bin/bash
#. ./path.sh

##
##
##
##  Perform Speech Activity Detection (SAD) based on neural network models trained on movies. 
##  Input must be a text file containing full paths to either mp4/mkv files or .wav 
##  files, and optionally the path to the directory where all the output will be 
##  stored (default=$proj_dir/sad_out_dir)
##  Output will be a timestamps file for each movie, each line of which will contain the 
##  start and end times for the speech segment.
##  Frame level posteriors are also saved.
##  Make sure to edit KALDI_ROOT path to point to complete path of Kaldi installation directory
##  before running this script
##  
##  E.g., bash perform_SAD.sh movie_path_list.txt expt_dir
##
##  Variables to be aware of:
##      out_dir       : directory where all the output-files will be created
##      nj            : number of jobs to be run in parallel 
##      feats_flag    : 'y' if you want to retain feature files after execution
##      wavs_flag     : 'y' if you want to retain '.wav' audio files after execution
##  
##  Packages/libraries required :
##     kaldi          : ensure that all kaldi binaries are added to system path. If not,
##                      either add them to system path, or modify KALDI_ROOT in 1st line of
##                      'path.sh' to reflect kaldi installation directory.
##     keras, tensorflow              :     required to load model and make SAD predictions.
##     re, gzip, struct               :     used by kaldi_io, to read feature files.
##

## Define usage of script
usage="Perform speech activity detection from audio
Usage: bash $(basename "$0") [-h] [-w y/n] [-f y/n] [-j num_jobs] movie_paths.txt (out_dir)
e.g.: bash $(basename "$0") -w y -f y -nj 8 demo.txt DEMO

where:
-h                  : Show help 
-w                  : Store wav files after processing (default: n)
-f                  : Store feature files after processing (default: n)
-j                 : Number of parallel jobs to run (default: 16)
movie_paths.txt     : Text file consisting of complete paths to media files (eg, .mp4/.mkv) on each line 
out_dir             : Directory in which to store all output files (default: "\$PWD"/SAD_out_dir)
"

## Kill all background processes if file exits due to error
trap "exit" INT TERM
trap "kill 0" EXIT
## Add kaldi binaries to path if path.sh file exists
if [ -f path.sh ]; then . ./path.sh; fi
## Default Options
feats_flag="n"
wavs_flag="n"
nj=16

## Input Options
if [ $# -eq 0 ];
then
    echo "$usage"
    exit
fi

while getopts ":hw:f:j:" option
do
    case "${option}"
    in
        h) echo "$usage"
        exit;;
        f) feats_flag="${OPTARG}";;
        w) wavs_flag="${OPTARG}";;
        j) nj=${OPTARG};;
        \?) echo "Invalid option: -$OPTARG" >&2 
        printf "See below for usage\n\n"
        echo "$usage"
        exit ;;
    esac
done
## Input Arguments
movie_list=${@:$OPTIND:1}
exp_id=$(($OPTIND+1))
if [ $# -ge $exp_id ]; then
    expt_dir=${@:$exp_id:1}
else
    expt_dir=SAD_out_dir
fi

## Reduce nj if not enough files
num_movies=`cat $movie_list | wc -l`
if [ $num_movies -lt $nj ]; then
    nj=$num_movies
fi

proj_dir=$(dirname "$0")
sad_model=$proj_dir/models/cnn_td.h5
wav_dir=$expt_dir/wavs
feats_dir=$expt_dir/features
scpfile=$feats_dir/wav.scp
lists_dir=$feats_dir/scp_lists
if [ -d "$expt_dir/SAD" ]; then rm -rf $expt_dir/SAD;fi
if [ -d "$wav_dir" ]; then rm -rf $wav_dir;fi
mkdir -p $wav_dir $feats_dir/log $lists_dir $expt_dir/SAD/{timestamps,posteriors} 

### Create .wav files given movie_files
echo " >>>> CREATING WAV FILES <<<< "
bash $proj_dir/inference_scripts/create_wav_files.sh $movie_list $wav_dir $nj
num_movies=`cat ${movie_list} | wc -l`
num_wav_extracted=`ls ${wav_dir} | wc -l`
if [ $num_movies -ne $num_wav_extracted ]; then
    echo "Unable to extract all .wav files, exiting..."
    exit 1
fi

### Extract fbank-features
echo " >>>> EXTRACTING FEATURES FOR SAD <<<< "
bash $proj_dir/inference_scripts/create_logmel_feats.sh $proj_dir $wav_dir $feats_dir $nj
num_feats=`cat ${feats_dir}/feats.scp | wc -l`
if [ $num_movies -ne $num_feats ]; then
    echo "Unable to extract all feature files, exiting..."
    exit 1
fi

## Generate SAD Labels
echo " >>>> GENERATING SAD LABELS <<<< "
movie_count=1
for movie_path in `cat $movie_list`
do
    movieName=`basename $movie_path | awk -F '.' '{print $1}'`
    cat $feats_dir/feats.scp | grep -- "${movieName}" > $lists_dir/${movieName}_feats.scp
    python $proj_dir/inference_scripts/model_predict.py $expt_dir $lists_dir/${movieName}_feats.scp $sad_model $nj & 
    if [ $(($movie_count % $nj)) -eq 0 ];then
        wait
    fi
    movie_count=`expr $movie_count + 1`
done
wait

## Delete feature files and/or wav files unless otherwise specified
if [[ "$feats_flag" == "n" ]]; then
    rm -r $feats_dir & 
fi
if [[ "$wavs_flag" == "n" ]]; then
    rm -r $wav_dir &
fi
wait
echo " >>>> SAD SEGMENTS CAN BE FOUND IN $expt_dir/SAD/timestamps <<<< "

