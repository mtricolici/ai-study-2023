
SCALE_FACTOR = 4
BATCH_SIZE   = 6

# 0.9 means 10% of dataset will not be used for traing
# it will be used for validation !
SPLIT_RATIO = 0.9

# Stop training If validation loss is not improving for EARLY_STOPPING_PATIENCE nr of epoches
# NR of epoches to wait before stopping after the model stops improving
# Common value between 5 .. 10
EARLY_STOPPING_PATIENCE = 5

NUM_RES_BLOCKS = 16
NUM_FILTERS    = 64

EPOCH = 5000 # It should stop anyway earlier ;)
STEPS_PER_EPOCH = 100
LEARNING_RATE = 0.001

MODEL_SAVE_PATH = '/output/edsr-model.keras'

# dataset directory should have files of type
# 1212-small.png
# 1212-big.png << scaled small image

DATASET_DIR = '/dataset'
SMALL_SUFFIX = 'small'
BIG_SUFFIX = 'big'

DEMO_INPUT_FILE = '/output/input.png'
DEMO_OUTPUT_FILE = '/output/scaled.png'

