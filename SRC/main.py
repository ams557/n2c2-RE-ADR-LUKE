from preprocessing import *

if __name__ == "__main__":
    df = BRATtoDFconvert(TRAINING_DIR)
    print(df)