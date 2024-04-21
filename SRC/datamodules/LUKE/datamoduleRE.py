import torch
from torch.nn import functional as f
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import os

class LUKE_RelationExtractionDataset(Dataset):
    
    def __init__(self, data, tokenizer, label2id):
        """
        Args:
            data : Pandas dataframe.
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
    def __init__(self, train_df, test_df, tokenizer, RelationExtractionDataset, valid_split, batch_size, num_workers, random_state, label2id):
        super().__init__()
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