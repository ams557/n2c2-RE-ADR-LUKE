from transformers import LukeForEntityPairClassification
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import torchmetrics
from experiments.n2c2.twenty18.task2.RE.config import WANDB_key
import wandb
wandb.login(key=WANDB_key)
import matplotlib.pyplot as plt


class LukeREmodel(pl.LightningModule):
    def __init__(self, learning_rate, id2label):
        super().__init__()
        self.id2label = id2label
        num_labels = len(self.id2label)
        self.learning_rate = learning_rate
        if num_labels is None:
            raise ValueError("must provide num_labels.")
        self.model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-base", num_labels=num_labels)
        self.criterion = torch.nn.CrossEntropyLoss()

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
        self.val_cm = torchmetrics.ConfusionMatrix(num_classes=num_labels,task="multiclass",normalize='true')

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
        self.val_precision_micro.update(preds,labels)
        self.val_recall_micro.update(preds,labels)
        self.val_F1_micro.update(preds,labels)
        self.val_precision_macro.update(preds,labels)
        self.val_recall_macro.update(preds,labels)
        self.val_F1_macro.update(preds,labels)
        self.val_cm.update(preds,labels)
        return loss
    
    def on_validation_epoch_end(self):
        val_P_class = self.val_precision.compute()
        val_R_class = self.val_recall.compute()
        val_F_class = self.val_F1.compute()
        for i, (F, P, R) in enumerate(zip(val_F_class,val_P_class,val_R_class)):
            self.log(f'val_F_class_{i}', F, on_epoch=True, prog_bar=True)
            self.log(f'val_P_class_{i}', P, on_epoch=True, prog_bar=True)
            self.log(f'val_R_class_{i}', F, on_epoch=True, prog_bar=True)
        self.log('val_accuracy_micro',self.val_accuracy_micro.compute())
        self.log('val_precision_micro',self.val_precision_micro.compute())
        self.log('val_recall_micro',self.val_recall_micro.compute())
        self.log('val_F1_micro',self.val_F1_micro.compute())
        self.log('val_accuracy_macro',self.val_accuracy_macro.compute())
        self.log('val_precision_macro',self.val_precision_macro.compute())
        self.log('val_recall_macro',self.val_recall_macro.compute())
        self.log('val_F1_macro',self.val_F1_macro.compute())
        fig,ax = self.val_cm.plot(add_text=True)
        wandb.log({'val_confusion_matrix' : [wandb.Image(fig)]})
        plt.close(fig)

        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_F1.reset()

        self.val_accuracy_micro.reset()
        self.val_precision_micro.reset()
        self.val_recall_micro.reset()
        self.val_F1_micro.reset()

        self.val_accuracy_macro.reset()
        self.val_precision_macro.reset()
        self.val_recall_macro.reset()
        self.val_F1_macro.reset()

        self.val_cm.reset()
    
    def test_step(self, batch, batch_idx):
        loss, preds, labels = self.common_step(batch,batch_idx)
        self.log('test_loss',loss)
        self.test_accuracy.update(preds,labels)
        self.test_precision.update(preds,labels)
        self.test_recall.update(preds,labels)
        self.test_F1.update(preds,labels)
        self.test_precision_micro.update(preds,labels)
        self.test_recall_micro.update(preds,labels)
        self.test_F1_micro.update(preds,labels)
        self.test_precision_macro.update(preds,labels)
        self.test_recall_macro.update(preds,labels)
        self.test_F1_macro.update(preds,labels)
        self.test_cm.update(preds,labels)
        return loss
    
    def on_test_epoch_end(self):
        test_P_class = self.test_precision.compute()
        test_R_class = self.test_recall.compute()
        test_F_class = self.test_F1.compute()
        for i, (F, P, R) in enumerate(zip(test_F_class,test_P_class,test_R_class)):
            self.log(f'test_F_class_{i}', F, on_epoch=True, prog_bar=True)
            self.log(f'test_P_class_{i}', P, on_epoch=True, prog_bar=True)
            self.log(f'test_R_class_{i}', F, on_epoch=True, prog_bar=True)
        self.log('test_accuracy_micro',self.test_accuracy_micro.compute())
        self.log('test_precision_micro',self.test_precision_micro.compute())
        self.log('test_recall_micro',self.test_recall_micro.compute())
        self.log('test_F1_micro',self.test_F1_micro.compute())
        self.log('test_accuracy_macro',self.test_accuracy_macro.compute())
        self.log('test_precision_macro',self.test_precision_macro.compute())
        self.log('test_recall_macro',self.test_recall_macro.compute())
        self.log('test_F1_macro',self.test_F1_macro.compute())
        fig,ax = self.test_cm.plot(add_text=True)
        wandb.log({'test_confusion_matrix' : [wandb.Image(fig)]})
        plt.close(fig)

        self.test_accuracy.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_F1.reset()

        self.test_accuracy_micro.reset()
        self.test_precision_micro.reset()
        self.test_recall_micro.reset()
        self.test_F1_micro.reset()

        self.test_accuracy_macro.reset()
        self.test_precision_macro.reset()
        self.test_recall_macro.reset()
        self.test_F1_macro.reset()
        
        self.test_cm.reset()

    def configure_optimizers(self):
        return AdamW(self.parameters(),lr=self.learning_rate)

