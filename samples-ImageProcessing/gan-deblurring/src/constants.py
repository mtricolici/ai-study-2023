
BATCH_SIZE   = 10

EPOCH = 1
STEPS_PER_EPOCH = 20
LEARNING_RATE = 0.001

MODEL_SAVE_PATH = '/output'
SAVE_BEST_ONLY = True

# dataset directory should have files of type
# good-1212.png << good quality image
# bad-1212.png  << bad  quality image (i.e. blurred)
# size of good and bad image pair MUST be the same!

DATASET_DIR = '/dataset'
GOOD_SUFFIX = 'good'
BAD_SUFFIX  = 'bad'

DEMO_INPUT_FILE = '/output/input.png'
DEMO_OUTPUT_FILE = '/output/result.png'

