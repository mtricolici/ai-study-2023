
BATCH_SIZE   = 5

EPOCH = 2
STEPS_PER_EPOCH = 200
LEARNING_RATE = 0.001

MODEL_SAVE_PATH = '/output'
SAVE_BEST_ONLY = True

# dataset directory should have files of type
# 2-good-1212.png << good quality image
# 2-bad-1212.png  << bad  quality image (i.e. blurred)
# size of good and bad image pair MUST be the same!
# first digit in file name is a index of resolution
# When we do predict with a batch - all images in a batch should be the same size!

DATASET_DIR = '/dataset'
RESOLUTIONS_COUNT = 5
GOOD_SUFFIX = 'good'
BAD_SUFFIX  = 'bad'

DEMO_INPUT_FILE = '/output/input.png'
DEMO_OUTPUT_FILE = '/output/result.png'

