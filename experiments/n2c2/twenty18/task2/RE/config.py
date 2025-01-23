# -*- coding: utf-8 -*-
"""File for setting up configuration variables

Example:
    To use this module, ensure the proper directory structre & import::

        >>> from experiments.n2c2.twenty18.task2.RE.config import *

Sources:
    * (LUKE Example) https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LUKE/Supervised_relation_extraction_with_LukeForEntityPairClassification.ipynb#scrollTo=hDkptorP9Koh
"""

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
WANDB_key = "" # your W&B API Key Here

# Training Hyperparameters
LEARNING_RATE = 1e-5
BATCH_SIZE = 4
MIN_EPOCHS = 1
MAX_EPOCHS = 6

# Dataset
DATA_DIR = 'datasets/n2c2/2018/task2/RE/'
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
TRAIN_DIR = DATA_DIR + 'train/'
TEST_DIR = DATA_DIR + 'test/'
VALID_SPLIT = 0.2
NUM_WORKERS = 4

# Compute related
RANDOM_STATE = 42
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = 16

# Logging
WANDB_LOGGER = WandbLogger(
    name='LUKE-N2C2-RE', project='LUKE'
)

# Model checkpointing
CALLBACKS = [
    ModelCheckpoint(
        save_top_k=1, 
        mode="max", 
        monitor="val_F1_macro", 
        save_last=True,
        dirpath='./logs/n2c2/2018/task2/RE/checkpoints/',
        auto_insert_metric_name=True,
        filename='{epoch}-{val_loss:.2f}-{val_F1_macro:.2f}'),
    EarlyStopping(monitor='val_loss',patience=2,strict=False,verbose=False,mode='min')
]