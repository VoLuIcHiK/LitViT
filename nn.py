import lightning as L
import torch.nn.functional as F
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from torch.optim.lr_scheduler import ExponentialLR, OneCycleLR
from vit_model.vit_from_scratch import ViT

class LitViT(L.LightningModule):
    def __init__(self, num_classes, len_train, batch_size, epochs, lr=1e-3):
        super().__init__()
        self.model = ViT(num_classes=num_classes)
        self.lr = lr
        self.num_classes = num_classes
        self.len_train = len_train
        self.batch_size = batch_size
        self.epochs = epochs

    def training_step(self, batch):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #lr_scheduler = ExponentialLR(optimizer, gamma=0.9)
        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=0.01,
            steps_per_epoch=2 * self.len_train,
            epochs=self.epochs,
            three_phase=True
        )
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]