DATA_DIR = '../DATA/'
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
TRAIN_DIR = DATA_DIR + 'train/'
Path(TRAIN_DIR).mkdir(parents=True, exist_ok=True)
TEST_DIR = DATA_DIR + 'test/'
Path(TEST_DIR).mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42