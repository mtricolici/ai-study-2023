
DYNAMIC_MODEL = True

INPUT_SIZE   = (160, 90)
SCALE_FACTOR = 4
BATCH_SIZE   = 10

EPOCH = 20
STEPS_PER_EPOCH = 500
LEARNING_RATE = 0.001

MODEL_SAVE_PATH = '/output/edsr-model.keras'
SAVE_BEST_ONLY = True

# dataset directory should have files of type
# 1212-small.png
# 1212-big.png << scaled small image

DATASET_DIR = '/dataset'
SMALL_SUFFIX = 'small'
BIG_SUFFIX = 'big'

DEMO_INPUT_FILE = '/output/input.png'
DEMO_OUTPUT_FILE = '/output/scaled.png'

