from transformers import LukeForEntityPairClassification
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import torchmetrics
from sklearn.model_selection import train_test_split
from config import *
from dataset import *
import wandb
import matplotlib.pyplot as plt

class LUKE(pl.LightningModule):
    def __init__(self,num_labels,learning_rate,dataset,tokenizer,label2id):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.train_ds, self.valid_ds,self.test_ds = self.create_datasets(self.dataset,self.tokenizer,self.label2id)
        self.learning_rate = learning_rate
        self.model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-base", num_labels=num_labels)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass',num_classes=num_labels)
        self.val_precision = torchmetrics.Precision(task='multiclass',num_classes=num_labels)
        self.val_F1 = torchmetrics.F1Score(task='multiclass',num_classes=num_labels)
        self.val_recall = torchmetrics.Recall(task='multiclass',num_classes=num_labels)
        self.val_F1_micro = torchmetrics.F1Score(task='multiclass',num_classes=num_labels,average='micro')
        self.val_F1_macro = torchmetrics.F1Score(task='multiclass',num_classes=num_labels,average='macro')
        self.val_cm = torchmetrics.ConfusionMatrix(num_classes=num_labels,task="multiclass",normalize='true')
        self.test_accuracy = torchmetrics.Accuracy(task='multiclass',num_classes=num_labels)
        self.test_precision = torchmetrics.Precision(task='multiclass',num_classes=num_labels)
        self.test_recall = torchmetrics.Recall(task='multiclass',num_classes=num_labels)
        self.test_F1 = torchmetrics.F1Score(task='multiclass',num_classes=num_labels)
        self.test_F1_micro = torchmetrics.F1Score(task='multiclass',num_classes=num_labels,average='micro')
        self.test_F1_macro = torchmetrics.F1Score(task='multiclass',num_classes=num_labels,average='macro')
        self.test_cm = torchmetrics.ConfusionMatrix(num_classes=num_labels,task="multiclass",normalize='true')

    def forward(self, input_ids, entity_ids, entity_position_ids, attention_mask, entity_attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, entity_ids=entity_ids, 
                            entity_attention_mask=entity_attention_mask, entity_position_ids=entity_position_ids)

    def common_step(self,batch,batch_idx):
        labels = batch['label']
        del batch['label']
        outputs = self(**batch)
        logits = outputs.logits
        preds = logits.argmax(-1)
        loss = self.criterion(logits,labels)
        return loss, preds, labels

    def training_step(self,batch,batch_idx):
        loss, preds, labels = self.common_step(batch,batch_idx)
        self.log("training_loss",loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.common_step(batch,batch_idx)
        self.log('val_loss',loss)
        self.val_accuracy.update(preds,labels)
        self.val_precision.update(preds,labels)
        self.val_recall.update(preds,labels)
        self.val_F1.update(preds,labels)
        self.val_F1_macro.update(preds,labels)
        self.val_F1_micro.update(preds,labels)
        self.val_cm.update(preds,labels)
        return loss
    
    def on_validation_epoch_end(self):
        self.log('val_accuracy',self.val_accuracy.compute())
        self.log('val_precision',self.val_precision.compute())
        self.log('val_recall',self.val_recall.compute())
        self.log('val_F1',self.val_F1.compute())
        self.log('val_F1_micro',self.val_F1_micro.compute())
        self.log('val_F1_macro',self.val_F1_macro.compute())
        fig,ax = self.val_cm.plot(add_text=False)
        wandb.log({'val_confusion_matrix' : [wandb.Image(fig)]})
        plt.close(fig)

        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_F1.reset()
        self.val_F1_micro.reset()
        self.val_F1_macro.reset()
        self.val_cm.reset()
    
    def test_step(self, batch, batch_idx):
        loss, preds, labels = self.common_step(batch,batch_idx)
        self.log('test_loss',loss)
        self.test_accuracy.update(preds,labels)
        self.test_precision.update(preds,labels)
        self.test_recall.update(preds,labels)
        self.test_F1.update(preds,labels)
        self.test_F1_macro.update(preds,labels)
        self.test_F1_micro.update(preds,labels)
        self.test_cm.update(preds,labels)
        return loss
    
    def on_test_epoch_end(self):
        self.log('test_accuracy',self.test_accuracy.compute())
        self.log('test_precision',self.test_precision.compute())
        self.log('test_recall',self.test_recall.compute())
        self.log('test_F1',self.test_F1.compute())
        self.log('test_F1_micro',self.test_F1_micro.compute())
        self.log('test_F1_macro',self.test_F1_macro.compute())
        fig,ax = self.test_cm.plot(add_text=False)
        wandb.log({'test_confusion_matrix' : [wandb.Image(fig)]})
        plt.close(fig)

        self.test_accuracy.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_F1.reset()
        self.test_F1_micro.reset()
        self.test_F1_macro.reset()
        self.test_cm.reset()
    
    def create_datasets(self,dataset,tokenizer,label2id):
        train_df, val_df = train_test_split(dataset['train'], test_size=0.2, random_state=RANDOM_STATE, shuffle=True)
        test_df = dataset['test']
        train_ds = RelationExtractionDataset(data=train_df,tokenizer=tokenizer,label2id=label2id)
        valid_ds = RelationExtractionDataset(data=val_df,tokenizer=tokenizer,label2id=label2id)
        test_ds = RelationExtractionDataset(data=test_df,tokenizer=tokenizer,label2id=label2id)
        return train_ds, valid_ds, test_ds

    def configure_optimizers(self):
        return AdamW(self.parameters(),lr=self.learning_rate)
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=4, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=2)
    def test_dataloader(self):
        return DataLoader(self.test_ds,batch_size=2)

