import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LukeTokenizer

class RelationExtractionDataset(Dataset):
    """Relation extraction dataset."""

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