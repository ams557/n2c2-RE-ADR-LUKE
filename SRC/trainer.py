import torch
import pytorch_lightning as pl
from src.models.pretrained.LUKE import LukeREmodel
from src.datamodules.LUKE.datamoduleRE import LUKE_RelationExtractionDataset, LUKE_N2C22018_Task2_RE_DataModule
from experiments.n2c2.twenty18.task2.RE.config import TRAIN_DIR, TEST_DIR, VALID_SPLIT, BATCH_SIZE, NUM_WORKERS, RANDOM_STATE, LEARNING_RATE, ACCELERATOR, DEVICES, MIN_EPOCHS, MAX_EPOCHS, WANDB_LOGGER, EARLY_STOPPING_CALLBACK
from transformers import LukeTokenizer
from src.preprocessing import *



if __name__ == "__main__":

    dataset = {
        'train' : BRATtoDFconvert(TRAIN_DIR),
        'test' : BRATtoDFconvert(TEST_DIR)
    }
    
    id2label = dict()
    for idx, label in enumerate(dataset['train'].string_id.value_counts().index):
        id2label[idx] = label
    label2id = {v:k for k,v in id2label.items()}
    num_labels = len(label2id)

    tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base", task="entity_pair_classification")

    dm = LUKE_N2C22018_Task2_RE_DataModule(
        train_df=dataset['train'],
        test_df=dataset['test'],
        tokenizer=tokenizer,
        RelationExtractionDataset=LUKE_RelationExtractionDataset,
        valid_split=VALID_SPLIT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        random_state=RANDOM_STATE,
        label2id=label2id
    )

    model = LukeREmodel(
        learning_rate=LEARNING_RATE,
        id2label=id2label
    )

    trainer = pl.Trainer(
        accelerator=ACCELERATOR,
        devices=DEVICES,
        min_epochs=MIN_EPOCHS,
        max_epochs=MAX_EPOCHS,
        logger=WANDB_LOGGER,
        callbacks=[EARLY_STOPPING_CALLBACK]
    )

    trainer.fit(model,dm)
    trainer.validate(model,dm)
    trainer.test(model,dm)

