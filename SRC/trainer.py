# -*- coding: utf-8 -*-
"""Module for running the model (includes testing & training) 

Example:
    To use this module, ensure the proper directory structure & run...

        $ python -m src.trainer.py

Sources:
    * (LUKE Example) https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LUKE/Supervised_relation_extraction_with_LukeForEntityPairClassification.ipynb#scrollTo=hDkptorP9Koh
    * (RELEX) https://github.com/NLPatVCU/RelEx/tree/master/relex
    * (Lightning) https://lightning.ai/docs/pytorch/stable/
"""

import torch
import pytorch_lightning as pl
from src.models.pretrained.LUKE import LukeREmodel
from src.datamodules.LUKE.datamoduleRE import LUKE_RelationExtractionDataset, LUKE_N2C22018_Task2_RE_DataModule
from experiments.n2c2.twenty18.task2.RE.config import TRAIN_DIR, TEST_DIR, VALID_SPLIT, BATCH_SIZE, NUM_WORKERS, RANDOM_STATE, LEARNING_RATE, ACCELERATOR, DEVICES, MIN_EPOCHS, MAX_EPOCHS, WANDB_LOGGER, CALLBACKS
from transformers import LukeTokenizer
from src.preprocessing import *
from src.utils.shared_utils import plot_json_logger # not working & excluded


if __name__ == "__main__":

    # create train & test datasets
    datasets = {
        'train' : BRATtoDFconvert(TRAIN_DIR),
        'test' : BRATtoDFconvert(TEST_DIR)
    }
    
    # create dictionary of ids & their paired labels (taken from LUKE Example)
    id2label = dict()
    for idx, label in enumerate(datasets['train'].string_id.value_counts().index):
        id2label[idx] = label
    
    # reverse items in dictionary (taken from LUKE Example)
    label2id = {v:k for k,v in id2label.items()}
    num_labels = len(label2id)

    # initialize tokenizer
    tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base", task="entity_pair_classification")

    # initialize datamodule
    dm = LUKE_N2C22018_Task2_RE_DataModule(
        train_df=datasets['train'],
        test_df=datasets['test'],
        tokenizer=tokenizer,
        RelationExtractionDataset=LUKE_RelationExtractionDataset,
        valid_split=VALID_SPLIT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        random_state=RANDOM_STATE,
        label2id=label2id
    )

    # initialize model
    model = LukeREmodel(
        learning_rate=LEARNING_RATE,
        id2label=id2label
    )

    # initialize trainer
    trainer = pl.Trainer(
        accelerator=ACCELERATOR,
        devices=DEVICES,
        min_epochs=MIN_EPOCHS,
        max_epochs=MAX_EPOCHS,
        logger=WANDB_LOGGER,
        callbacks=CALLBACKS
    )

    # trainin & validating the model
    trainer.fit(model=model,datamodule=dm)

    # testing on the test dataset
    trainer.test(model=model,datamodule=dm,ckpt_path="best")

