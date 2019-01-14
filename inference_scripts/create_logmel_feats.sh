#!/bin/bash
## 
##
##  Extract 64D log-Mel filterbank energy features
##
##  Arguments :
##
##      proj_dir    - Directory containing the scripts
##      wav_dir     - Directory in which .wav files are stored
##      feats_dir   - Directory in which to store features
##      nj          - Number of parallel jobs to run
##

proj_dir=$1
wav_dir=$2
feats_dir=$3
nj=$4

log_dir=$feats_dir/log
fbank_dir=$feats_dir/fbank_data
scp=$feats_dir/wav.scp
path_file=$proj_dir/path.sh
mkdir -p $log_dir $fbank_dir 
if [ -f $path_file ];  then . $path_file; fi 

## Create 'wav.scp' file for kaldi feature extraction

find $wav_dir -type f -name '*.wav' | while read r
do
    movie_name=`basename $r .wav`
    echo $movie_name" "$r
done > $scp

sort $scp -o $scp

####
####    Extract log-Mel filterbank coefficients
####

## Split wav.scp into 'nj' parts
split_wav_scp=""
for n in $(seq $nj); do
    split_wav_scp="$split_wav_scp $log_dir/wav.scp.$n"
done
$proj_dir/inference_scripts/split_scp.pl $scp $split_wav_scp || exit 1;

## Extract fbank features using run.pl parallelization
$proj_dir/inference_scripts/run.pl JOB=1:$nj $log_dir/make_fbank_feats.JOB.log \
    compute-fbank-feats --verbose=2 --num-mel-bins=64 scp:$log_dir/wav.scp.JOB ark,p:- \| \
    copy-feats --compress=true ark,p:- \
        ark,scp,p:$fbank_dir/raw_fbank_feats.JOB.ark,$fbank_dir/raw_fbank_feats.JOB.scp \
|| exit 1;

## Combine multiple fbank files 
for n in $(seq $nj); do
  cat $fbank_dir/raw_fbank_feats.$n.scp || exit 1;
done > $feats_dir/feats.scp

