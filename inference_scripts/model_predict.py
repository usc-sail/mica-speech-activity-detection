###
###     Python Script to generate SAD labels given 
###     log-Mel features as input
###
###     INPUTS:
###     write_dir    -  Directory in which to write all output files
###     scp_file     -  Kaldi feature file in .scp format
###     model_file   -  SAD model 
###

###
###     OUTPUTS:
###     Frame-level posteriors from the model predictions
###     are thresholded at 0.5.
###    
###     write_post   -  Raw posteriors representing confidence in SAD prediction
###     write_ts     -  SAD segments detected written as start and end 
###                     end times.
###
###

import os, sys, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
np.warnings.filterwarnings('ignore')
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
from kaldi_io import read_mat_scp
from keras.models import load_model
from keras import backend as K


##
##  Convert frame-level posteriors into 
##  continuous segments of regions where post=pos_label
##
def frame2seg(frames, frame_time_sec=0.01, pos_label=1):
    pos_idxs = np.where(frames==pos_label)[0]
    pos_regions = np.split(pos_idxs, np.where(np.diff(pos_idxs)!=1)[0]+1)
    if len(pos_idxs) == 0 or len(pos_regions) == 0:
        return []
    segments = np.array([[x[0], x[-1]+1] for x in pos_regions])*frame_time_sec
    return segments

def normalize(data):
    return np.divide(np.subtract(data, np.mean(data)), np.std(data))

def main():
    write_dir, scp_file, model_file, nj = sys.argv[1:]
    K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))
    frame_len = 0.01
    write_post = os.path.join(write_dir, 'SAD/posteriors/')
    write_ts   = os.path.join(write_dir, 'SAD/timestamps/')

    model = load_model(model_file)
    gen = read_mat_scp(scp_file)

    # Generate SAD posteriors using pre-trained SAD model
    for movie, fts in gen:
        predictions = []
        num_seg = int(len(fts)//64)
        for i in range(num_seg):
            feats_seg = normalize(fts[i*64:(i+1)*64])
            pred = model.predict(feats_seg.reshape((1, 64, 64, 1)), verbose=0)
            predictions.append(pred[0][1])

        # Post-processing of posteriors
        labels = np.array([np.repeat(x, 64) for x in predictions]).flatten()
        seg_times = frame2seg(np.round(labels))
        # Write start and end SAD timestamps 
        fw = open(os.path.join(write_ts, movie + '.ts'),'w')

        seg_ct = 1
        for segment in seg_times:
            if segment[1]-segment[0] > 0.05:
                fw.write('{0}_sad-{1:04}\t{0}\t{2:0.2f}\t{3:0.2f}\n'.format(movie, seg_ct, segment[0], segment[1]))
                seg_ct += 1
        fw.close()

        # Write frame-level posterior probabilities
        fpost = open(os.path.join(write_post, movie + '.post'),'w')
        for frame in predictions:
            fpost.write('{0:0.2f}\n'.format(frame))
        fpost.close()

if __name__=='__main__':
    main()
