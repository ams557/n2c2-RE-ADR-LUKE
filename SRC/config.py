DATA_DIR = '../DATA/'
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
TRAIN_DIR = DATA_DIR + 'train/training_20180910/'
Path(TRAIN_DIR).mkdir(parents=True, exist_ok=True)
TEST_DIR = DATA_DIR + 'test/test_data_Task2/'
Path(TEST_DIR).mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42