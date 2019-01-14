# make sure the kaldi binary files are indicated in this path file
# set this manually
export KALDI_ROOT=/proj/tools/kaldi

export PATH=$PWD/utils/:$KALDI_ROOT/src/diarbin/:$KALDI_ROOT/src/featbin/:$PATH

[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
