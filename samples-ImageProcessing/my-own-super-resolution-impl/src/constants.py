
INPUT_SIZE   = (160, 90)
SCALE_FACTOR = 4
BATCH_SIZE   = 15

EPOCH = 1 #10
STEPS_PER_EPOCH = 3 # 500
LEARNING_RATE = 0.001

MODEL_SAVE_PATH = '/output/edsr-model.h5'
SAVE_BEST_ONLY = False

# dataset directory should have files of type
# small-1212.png
# big-1212.png << scaled small image

DATASET_DIR = '/dataset'
SMALL_SUFFIX = 'small'
BIG_SUFFIX = 'big'

DEMO_INPUT_FILE = '/output/input.png'
DEMO_OUTPUT_FILE = '/output/scaled.png'

