import torch

class RelationExtractionDataset(Dataset):
    """Relation extraction dataset."""

    def __init__(self, data):
        """
        Args:
            data : Pandas dataframe.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        sentences = item.sentences
        entity_spans = [tuple(x) for x in item.entity_spans]

        encoding = tokenizer(sentences, entity_spans=entity_spans, padding='max_length', truncation=True, return_tensors="pt",max_length=257)

        for k,v in encoding.items():
          encoding[k] = encoding[k].squeeze()

        encoding["label"] = torch.tensor(label2id[item.string_id])

        return encoding