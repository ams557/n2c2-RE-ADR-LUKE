from pathlib import Path

WANDB_key = "3f4a097a574f34b0356bb664fb479ba2c4217659"
DATA_DIR = '../DATA/'
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
TRAIN_DIR = DATA_DIR + 'train/'
Path(TRAIN_DIR).mkdir(parents=True, exist_ok=True)
TEST_DIR = DATA_DIR + 'test/'
Path(TEST_DIR).mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
LEARNING_RATE = 1e-5