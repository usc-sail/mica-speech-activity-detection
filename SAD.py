import os

##
## PATH-RELATED PARAMETERS
##
LOG_DIR = os.path.join('/enter/complete/path/to/dir/in/which/to/create/logs/dir', 'logs')
DATA_PATH = '/enter/complete/path/to/folder/containing/train/val/test/directories' 
DATA_PATH_SP_TRAIN = os.path.join(DATA_PATH, 'train/speech')    
DATA_PATH_NS_TRAIN = os.path.join(DATA_PATH, 'train/non-speech')
DATA_PATH_SP_VAL = os.path.join(DATA_PATH, 'val/speech')
DATA_PATH_NS_VAL = os.path.join(DATA_PATH, 'val/non-speech')
DATA_PATH_SP_TEST = os.path.join(DATA_PATH, 'test/speech')      
DATA_PATH_NS_TEST = os.path.join(DATA_PATH, 'test/non-speech')

###
### INPUT FEATURE PARAMETERS
###
FEAT_DIM = (128, 64)            # Dimension of features stored as .tfrecord files
INPUT_SHAPE = (64, 64, 1)       # Feature-dimension used in training (Time, Frequency, Channels)
NUM_VAL_SAMPLES = 3500          # Number of validation feature files per class
NUM_TEST_SAMPLES_SP = 9578      # Number of test 'speech' feature files
NUM_TEST_SAMPLES_NS = 35348     # Number of test 'non-speech' feature files

##
## TRAINING PARAMETERS
##
LEARNING_RATE = 1e-4    # Learning rate for Adam optimizer
NUM_EPOCHS = 50         # Number of epochs during training
BATCH_SIZE = 64         # Total batch size, must be a multiple of 4 (see data_loader.py)
NUM_STEPS = int(1e5/BATCH_SIZE)     # Approx 50k training samples of minority class
PATIENCE = 3            # Number of epochs to wait after validation loss stops improving
LOG_FREQ  = 100         # Frequency of batches to log training metrics during an epoch
GPU_FRAC = 0.5          # Fraction of GPU to be used
