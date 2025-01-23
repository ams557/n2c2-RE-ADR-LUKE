from transformers import LukeForEntityPairClassification
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import torchmetrics
from experiments.n2c2.twenty18.task2.RE.config import WANDB_key
import wandb
import numpy as np
wandb.login(key=WANDB_key)
import matplotlib.pyplot as plt


class LukeREmodel(pl.LightningModule):
    def __init__(self, learning_rate, id2label):
        super().__init__()
        self.id2label = id2label
        if len(self.id2label) == 0:
            raise ValueError("must provide num_labels.")
        num_labels = len(self.id2label)
        self.learning_rate = learning_rate
        self.model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-base", num_labels=num_labels)
        self.criterion = torch.nn.CrossEntropyLoss()

        # Training
        self.train_F1_micro = torchmetrics.F1Score(task='multiclass',num_classes=num_labels,average='micro')
        self.train_F1_macro = torchmetrics.F1Score(task='multiclass',num_classes=num_labels,average='macro')

        # Validation
        ## Non-Averaged
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass',num_classes=num_labels,average=None)
        self.val_precision = torchmetrics.Precision(task='multiclass',num_classes=num_labels,average=None)
        self.val_recall = torchmetrics.Recall(task='multiclass',num_classes=num_labels,average=None)
        self.val_F1 = torchmetrics.F1Score(task='multiclass',num_classes=num_labels, average=None)

        ## System - Micro-Averaged
        self.val_accuracy_micro = torchmetrics.Accuracy(task='multiclass',num_classes=num_labels,average='micro')
        self.val_precision_micro = torchmetrics.Precision(task='multiclass',num_classes=num_labels,average='micro')
        self.val_recall_micro = torchmetrics.Recall(task='multiclass',num_classes=num_labels,average='micro')
        self.val_F1_micro = torchmetrics.F1Score(task='multiclass',num_classes=num_labels,average='micro')

        ## System - Macro-Averaged
        self.val_accuracy_macro = torchmetrics.Accuracy(task='multiclass',num_classes=num_labels,average='macro')
        self.val_precision_macro = torchmetrics.Precision(task='multiclass',num_classes=num_labels,average='macro')
        self.val_recall_macro = torchmetrics.Recall(task='multiclass',num_classes=num_labels,average='macro')
        self.val_F1_macro = torchmetrics.F1Score(task='multiclass',num_classes=num_labels,average='macro')

        ## Confusion Matrix
        self.val_cm = torchmetrics.ConfusionMatrix(num_classes=num_labels,task="multiclass")

        # Test
        self.test_accuracy = torchmetrics.Accuracy(task='multiclass',num_classes=num_labels,average=None)
        self.test_precision = torchmetrics.Precision(task='multiclass',num_classes=num_labels,average=None)
        self.test_recall = torchmetrics.Recall(task='multiclass',num_classes=num_labels,average=None)
        self.test_F1 = torchmetrics.F1Score(task='multiclass',num_classes=num_labels, average=None)

        ## System - Micro-Averaged
        self.test_accuracy_micro = torchmetrics.Accuracy(task='multiclass',num_classes=num_labels,average='micro')
        self.test_precision_micro = torchmetrics.Precision(task='multiclass',num_classes=num_labels,average='micro')
        self.test_recall_micro = torchmetrics.Recall(task='multiclass',num_classes=num_labels,average='micro')
        self.test_F1_micro = torchmetrics.F1Score(task='multiclass',num_classes=num_labels,average='micro')

        ## System - Macro-Averaged
        self.test_accuracy_macro = torchmetrics.Accuracy(task='multiclass',num_classes=num_labels,average='macro')
        self.test_precision_macro = torchmetrics.Precision(task='multiclass',num_classes=num_labels,average='macro')
        self.test_recall_macro = torchmetrics.Recall(task='multiclass',num_classes=num_labels,average='macro')
        self.test_F1_macro = torchmetrics.F1Score(task='multiclass',num_classes=num_labels,average='macro')

        ## Confusion Matrix
        self.test_cm = torchmetrics.ConfusionMatrix(num_classes=num_labels,task="multiclass")

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
        self.log("training_loss_per_epoch",loss,on_epoch=True,on_step=False)
        self.train_F1_macro(preds,labels)
        self.log(
            "train_F1_macro", self.train_F1_macro,prog_bar=True,on_epoch=True, on_step=False
        )
        self.train_F1_micro(preds,labels)
        self.log(
            "train_F1_micro", self.train_F1_micro, prog_bar=True, on_epoch=True, on_step=False
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.common_step(batch,batch_idx)
        self.log('val_loss',loss, prog_bar=True)
        self.val_F1_micro(preds,labels)
        self.log(
            "val_F1_micro", self.val_F1_micro, prog_bar=True, on_epoch=True, on_step=False
        )
        self.val_F1_macro(preds,labels)
        self.log(
            "val_F1_macro", self.val_F1_macro,prog_bar=True,on_epoch=True, on_step=False
        )
        self.val_cm(preds,labels)
        self.val_preds = preds.detach().to('cpu').numpy()
        self.val_labels = labels.detach().to('cpu').numpy()
        return loss
    
    def on_validation_epoch_end(self):
        # FIXME: True HTML table vs image
        wandb.log({'val_confusion_matrix' : wandb.sklearn.plot_confusion_matrix(
            y_true=self.val_labels,
            y_pred=self.val_preds,
            labels=list(self.id2label.values())
        )})
        # fig, ax = self.val_cm.plot(add_text=True)
        # wandb.log({f'val_confusion_matrix': [wandb.Image(fig)]})
        # plt.close(fig)

    
    def test_step(self, batch, batch_idx):
        loss, preds, labels = self.common_step(batch,batch_idx)
        self.test_accuracy(preds,labels)
        self.test_precision(preds,labels)
        self.test_recall(preds,labels)
        self.test_F1(preds,labels)

        self.test_accuracy_micro(preds,labels)
        self.log(
            "test_accuracy_micro", self.test_accuracy_micro,prog_bar=True,on_epoch=True, on_step=False,
        )
        self.test_precision_micro(preds,labels)
        self.log(
            "test_precision_micro", self.test_precision_micro,prog_bar=True,on_epoch=True, on_step=False,
        )
        self.test_recall_micro(preds,labels)
        self.log(
            "test_recall_micro", self.test_recall_micro,prog_bar=True,on_epoch=True, on_step=False,
        )
        self.test_F1_micro(preds,labels)
        self.log(
            "test_F1_micro", self.test_F1_micro,prog_bar=True,on_epoch=True, on_step=False,
        )

        self.test_accuracy_macro(preds,labels)
        self.log(
            "test_accuracy_macro", self.test_accuracy_macro,prog_bar=True,on_epoch=True, on_step=False,
        )
        self.test_precision_macro(preds,labels)
        self.log(
            "test_precision_macro", self.test_precision_macro,prog_bar=True,on_epoch=True, on_step=False,
        )
        self.test_recall_macro(preds,labels)
        self.log(
            "test_recall_macro", self.test_recall_macro,prog_bar=True,on_epoch=True, on_step=False,
        )
        self.test_F1_macro(preds,labels)
        self.log(
            "test_F1_macro", self.test_F1_macro,prog_bar=True,on_epoch=True, on_step=False,
        )
        self.test_cm(preds,labels)
        self.test_preds = preds.detach().to('cpu').numpy()
        self.test_labels = labels.detach().to('cpu').numpy()
        return loss
    
    def on_test_epoch_end(self):
        test_P_class = self.test_precision.compute()
        test_R_class = self.test_recall.compute()
        test_F_class = self.test_F1.compute()
        for i, (F, P, R) in enumerate(zip(test_F_class,test_P_class,test_R_class)):
            self.log(f'test_F_class_{i}', F, on_epoch=True, prog_bar=True)
            self.log(f'test_P_class_{i}', P, on_epoch=True, prog_bar=True)
            self.log(f'test_R_class_{i}', R, on_epoch=True, prog_bar=True)
        # FIXME: Improve logging of CM in W&B
        wandb.log({'test_confusion_matrix' : wandb.sklearn.plot_confusion_matrix(
            y_true=self.test_labels,
            y_pred=self.test_preds,
            labels=list(self.id2label.values())
        )})
        # fig, ax = self.test_cm.plot(add_text=True)
        # wandb.log({f'test_confusion_matrix': [wandb.Image(fig)]})
        # plt.close(fig)

    def configure_optimizers(self):
        return AdamW(self.parameters(),lr=self.learning_rate)

