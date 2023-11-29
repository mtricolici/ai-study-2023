
BATCH_SIZE = 5

# Stop training If validation loss is not improving for EARLY_STOPPING_PATIENCE nr of epoches
# NR of epoches to wait before stopping after the model stops improving
# Common value between 5 .. 10
EARLY_STOPPING_PATIENCE = 10

EPOCH = 1000
STEPS_PER_EPOCH = 100
LEARNING_RATE = 0.01

MODEL_SAVE_PATH = '/output/my-cnn.keras'

DATASET_DIR = '/dataset'

DEMO_INPUT_FILE = '/output/input.png'
DEMO_OUTPUT_FILE = '/output/result.png'

