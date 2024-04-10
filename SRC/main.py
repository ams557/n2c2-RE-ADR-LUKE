from config import *
from preprocessing import *
from dataset import *
from model import *
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from transformers import LukeTokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl
import wandb
wandb.login(key="3f4a097a574f34b0356bb664fb479ba2c4217659")

if __name__ == "__main__":
    df = BRATtoDFconvert(TRAINING_DIR)
    id2label = dict()
    for idx, label in enumerate(train_df.string_id.value_counts().index):
        id2label[idx] = label
    label2id = {v:k for k,v in id2label.items()}
    tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base", task="entity_pair_classification")
    train_df, val_df = train_test_split(train_df_load, test_size=0.2, random_state=RANDOM_STATE, shuffle=True)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=RANDOM_STATE, shuffle=True)
    train_dataset = RelationExtractionDataset(data=train_df)
    valid_dataset = RelationExtractionDataset(data=val_df)
    # test_dataset = RelationExtractionDataset(data=test_df)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=2)
    # test_dataloader = DataLoader(test_dataset, batch_size=2)
    batch = next(iter(train_dataloader))
    tokenizer.decode(batch["input_ids"][1])
    batch = next(iter(valid_dataloader))
    labels = batch["label"]
    model = LUKE()
    del batch["label"]
    outputs = model(**batch)
    criterion = torch.nn.CrossEntropyLoss()
    initial_loss = criterion(outputs.logits, labels)
    print("Initial loss:", initial_loss)
    wandb_logger = WandbLogger(name='luke-first-run-12000-articles-bis', project='LUKE')
    # for early stopping, see https://pytorch-lightning.readthedocs.io/en/1.0.0/early_stopping.html?highlight=early%20stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=2,
        strict=False,
        verbose=False,
        mode='min'
    )
    trainer = Trainer(logger=wandb_logger, callbacks=[EarlyStopping(monitor='validation_loss')])
    trainer.fit(model)