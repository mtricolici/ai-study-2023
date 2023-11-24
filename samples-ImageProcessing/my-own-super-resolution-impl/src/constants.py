
SCALE_FACTOR = 4
BATCH_SIZE   = 10
VALIDATION_STEPS = 17 # !!! Number of validation samples / BATCH_SIZE

NUM_RES_BLOCKS = 16
NUM_FILTERS    = 64

EPOCH = 2
STEPS_PER_EPOCH = 50
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

