import os


###
###     ark_to_feats : Segment ark_file features to movie-wise
###     feature files, with each line consisting frame-wise features
###     of length feat_dim followed by the label for the frame.
###

def ark_to_feats_labels( ark_path, feat_path, label_path, feat_dim ):
    for files in os.listdir(ark_path):
        if files.endswith('ark'):
            fark=open(ark_path+files,'r')
            line_num=0
            data_line=fark.readline()
            while(data_line!=''):
                data=data_line.strip().split()
                if(len(data)==2):   # First line of every movie in .ark file
                    movie=data[0]
                    print(movie)
                    fw=open(feat_path+movie,'w')
                    flabel=open(label_path+movie,'r').read().split()
                    frm=0
                elif(len(data)==feat_dim):     # Every feature line
                    fw.write(data_line.rstrip())
                    fw.write(' '+flabel[frm]+'\n')
                    frm+=1
                else:                     # End of movie
                    fw.write(data_line.rstrip())
                    fw.write(' '+flabel[frm]+'\n')
                    fw.close()
                data_line=fark.readline()


###
###     ark_to_prob  : Script to write output labels which are stored
###     in .ark format(E.g, output of nnet-forward in kaldi)
###     Here, the output layer has 2 softmax-activated nodes,
###     of which we are writing the second output (P[X=1])
###     to file
###

def ark_to_prob( ark_path, prob_path, num_nodes ):
    for files in os.listdir(ark_path):
        if files.endswith('ark'):
            data=[x.strip().split() for x in open(ark_path+files,'r').readlines()]
            line_num=0
            while(line_num<len(data)):
                movie_name=data[line_num][0]
                line_num+=1
                fw=open(prob_path+movie_name,'w')
                while(len(data[line_num]) != num_nodes+1):
                    for ind in range(num_nodes):
                        fw.write(data[line_num][ind]+' ')
                    fw.write('\n')
                    line_num+=1
                for ind in range(num_nodes):
                    fw.write(data[line_num][ind]+' ')
                fw.close()
                line_num+=1


if __name__ == '__main__':
    test_dir='/proj/rajat/keras_model/train/'
    ark_path=test_dir+'mfcc_data/'
    feat_path=test_dir+'feats/'
    label_path='/proj/rajat/movie_data/labels/'
    feat_dim = 403
    ark_to_feats(ark_path, feat_path, label_path, feat_dim)
    
    test_dir='/proj/rajat/kaldi/Adapt_RATS_VAD/adapt_feats/test/'
    ark_path=test_dir+'decoded_newl/'
    prob_path=test_dir+'decoded_newl/vad_prob/'
    num_output_nodes = 2
    ark_to_prob(ark_path, prob_path, num_output_nodes)
    
