#!/bin/bash

# Copyright 2012-2016  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
START=$(date +%s)
nj=3
cmd=run.pl
mfcc_config=mfcc_NbCeps13.conf
compress=true
write_utt2num_frames=false  # if true writes utt2num_frames
splice_ctx=15
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
   echo "Usage: $0 [options] <data-dir> [<log-dir> [<mfcc-dir>] ]";
   echo "e.g.: $0 test_data logs/splice_feats test_data/mfcc_data"
   echo "Note: <log-dir> defaults to <data-dir>/log, and <mfccdir> defaults to <data-dir>/data"
   echo "Options: "
   echo "  --mfcc-config <config-file>                      # config passed to compute-mfcc-feats "
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --write-utt2num-frames <true|false>     # If true, write utt2num_frames file."
   exit 1;
fi


data=$1
if [ $# -ge 2 ]; then
  logdir=$2
else
  logdir=$data/log
fi
if [ $# -ge 3 ]; then
  mfccdir=$3
else
  mfccdir=$data/data
fi


# make $mfccdir an absolute pathname.
mfccdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $mfccdir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

utils/split_data.sh $data $nj
# add ,p to the input rspecifier so that we can just skip over
# utterances that have bad wave data.

$cmd JOB=1:$nj $logdir/splice_mfcc_${name}.JOB.log \
  splice-feats --left-context=$splice_ctx --right-context=$splice_ctx \
   scp:$data/split$nj/JOB/feats.scp\
   ark:- \| \
   subsample-feats --n=5 ark:- \
   ark,scp,t:$mfccdir/splice_mfcc_ss_$name.JOB.ark,$mfccdir/splice_mfcc_ss_$name.JOB.scp \
    || exit 1;


# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $mfccdir/splice_mfcc_ss_$name.$n.scp || exit 1;
done > $data/tst_splice_ss.scp || exit 1

rm -r $data/split$nj
echo "Succeeded splicing MFCC features for $name"
END=$(date +%s)
DIFF=$((END-START))
echo "splice_feats.sh took $DIFF seconds"
