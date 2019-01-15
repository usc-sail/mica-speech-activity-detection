# mica-sad-experiments
Speech Activity Detection (SAD) in movie audio. 

# Training
The details of the Subtitle-Aligned Movie (SAM) Corpus are provided in the [wiki](https://github.com/usc-sail/mica-vad-experiments/wiki)
In order to train an SAD model from scratch using this data, the following steps have to be performed after cloning the directory:
1. Download the features from here.
2. Extract the features : **tar -xvzf movie_features.tar.gz /path/to/dl/**
3. In the file SAD_parameters.py, edit DATA_PATH to reflect complete path to where the data is stored, and LOG_DIR to reflect directory in which to save logs and model file.
4. Additionally, training parameters can be modified.

To train a model, execute the train.py script \
**python train.py**

# Inference 
Since we use log-mel filterbank features extracted using Kaldi, a Kaldi installation is necessary to run the SAD inference script. Kaldi can be installed by following the instructions [here](https://github.com/kaldi-asr/kaldi). 
Once Kaldi is installed, make sure to edit the KALDI_ROOT variable in the 'path.sh' file to reflect the complete path to the installation directory. 
SAD can be performed on any media file compatible with ffmpeg (recommended types .mp4/.mkv/.wav) by executing the script 'perform_SAD.sh'.



Perform speech activity detection from audio\
Usage: bash perform_SAD.sh [-h] [-w y/n] [-f y/n] [-j num_jobs] movie_paths.txt (out_dir)\
e.g.: bash perform_SAD.sh -w y -f y -nj 8 demo.txt DEMO\
where:\
-h                  : Show help \
-w                  : Store wav files after processing (default: n)\
-f                  : Store feature files after processing (default: n)\
-j                  : Number of parallel jobs to run (default: 16)\
movie_paths.txt     : Text file consisting of complete paths to media files (eg, .mp4/.mkv) on each line \
out_dir             : Directory in which to store all output files (default: "\$PWD"/SAD_out_dir)\
