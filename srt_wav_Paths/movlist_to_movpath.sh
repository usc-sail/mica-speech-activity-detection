#!/bin/bash

if [ $# -eq 1 ]; then
    write_path={1}
else
    write_path=./movpath.txt
fi
testlist=/proj/rajat/keras_model/gentle_aligned_data/testlist
movPath=/proj/rajat/srt_wav_Paths/all_srt_movie_list.txt

while read -r movie; do
    movie_path=`grep "$movie" $movPath | awk -F ',' '{ print $2 }'`
    printf "$movie_path\n" >> $write_path
done < $testlist
