# -*- coding: utf-8 -*-
"""Module for setting up lightning datamodules

Example:
    To use this module, ensure the proper directory structure & import it...

        >>> from src.datamodules.LUKE.datamoduleRE import *

Sources:
    * (LUKE Example) https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LUKE/Supervised_relation_extraction_with_LukeForEntityPairClassification.ipynb#scrollTo=hDkptorP9Koh
    * (RELEX) https://github.com/NLPatVCU/RelEx/tree/master/relex
    * (Lightning) https://lightning.ai/docs/pytorch/stable/
"""

import torch
from torch.nn import functional as f
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import os

class LUKE_RelationExtractionDataset(Dataset):
    """Structure to set up dataframe for LUKE for entity pair classification

    Args:
        Dataset: Input dataset
    """
    def __init__(self, data, tokenizer, label2id):
        """Initialize variables for the class.

        Args:
            data: Pandas dataframe with information needed for LUKE for Entity Pair Classification Model.
            tokenizer: Model tokenizer
            label2id: Dictionary of labels & corresponding ids

        - Adapted from LUKE Example
        """
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        sentences = item.sentences
        entity_spans = [tuple(x) for x in item.entity_spans]

        encoding = self.tokenizer(sentences, entity_spans=entity_spans, padding='max_length', truncation=True, return_tensors="pt")

        for k,v in encoding.items():
          encoding[k] = encoding[k].squeeze()

        encoding["label"] = torch.tensor(self.label2id[item.string_id])

        return encoding

class LUKE_N2C22018_Task2_RE_DataModule(pl.LightningDataModule):
    """Pytorch datamodule for LUKE RE for N2C2 2018 Task 2 ADR-RE

    Args:
        pl.LightningDataModule: The LightningDataModule class to inherit from

    - Adapted from Lightning documentation & LUKE Example
    """

    def __init__(self, train_df, test_df, tokenizer, RelationExtractionDataset, valid_split, batch_size, num_workers, random_state, label2id):
        """Initialize class variables::

        Args:
            train_df: Training dataframe
            test_df: Testing dataframe
            tokenizer: Tokenizer
            RelationExtractionDataset: Dataset setup for relation extraction task
            valid_split: Validation set proportion
            batch_size: Batch size
            num_workers: Number of worker processes
            random_state: Random state
            label2id: Dictionary of labels & their corresponding ids
        """
        super().__init__() # inherit from LightningDataModule class
        self.train_df = train_df
        self.test_df = test_df
        self.RelationExtractionDataset = RelationExtractionDataset
        self.valid_split = valid_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state
        self.tokenizer = tokenizer
        self.label2id = label2id
        
    def setup(self,stage: str):
        """Setting up training & validation splits & converting formats

        Args:
            stage (string): Stage that trainer is in (not used)
        """
        train_df, val_df = train_test_split(self.train_df, test_size=self.valid_split, random_state=self.random_state, shuffle=True)
        self.train_ds = self.RelationExtractionDataset(data=train_df,tokenizer=self.tokenizer,label2id=self.label2id)
        self.valid_ds = self.RelationExtractionDataset(data=val_df,tokenizer=self.tokenizer,label2id=self.label2id)
        self.test_ds = self.RelationExtractionDataset(data=self.test_df,tokenizer=self.tokenizer,label2id=self.label2id)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers)
    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=int(self.batch_size/2),num_workers=self.num_workers)
    def test_dataloader(self):
        return DataLoader(self.test_ds,batch_size=int(self.batch_size/2),num_workers=self.num_workers)