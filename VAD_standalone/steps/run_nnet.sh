#!/bin/bash

# Copyright 2012-2016  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
START=$(date +%s)
nj=3
cmd=run.pl
compress=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 5 ]; then
   echo "Usage: $0 [options] <NN.nnet> <log-dir> <mfcc-dir> <dec-dir>"
   echo "e.g.: $0 NN/final.nnet logs/decoded test_data/mfcc_data decoded"
   echo "Options: "
   echo "  --mfcc-config <config-file>                      # config passed to compute-mfcc-feats "
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

NN=$1
logdir=$2
mfccdir=$3
decoded=$4
data=$5

# make $mfccdir an absolute pathname.
mfccdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $mfccdir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

$cmd JOB=1:$nj $logdir/vad_prob_log_${name}.JOB.ark \
    nnet-forward $NN scp:$mfccdir/splice_mfcc_$name.JOB.scp \
    ark,scp,t:$decoded/vad_prob.JOB.ark,$decoded/vad_prob.JOB.scp\
    || exit 1;


# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $decoded/vad_prob.$n.scp || exit 1;
done > $decoded/vad_prob.scp || exit 1

echo "Succeeded testing Neural Network"
END=$(date +%s)
DIFF=$((END-START))
echo "run_nnet.sh took $DIFF seconds"
