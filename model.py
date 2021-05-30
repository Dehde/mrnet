import torch
import torch.nn as nn
from torchvision import models
import pytorch_lightning as pl
from torch.nn import functional as F
import torchmetrics

class MRNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.alexnet(pretrained=True)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Linear(256, 3)
        self.sigmoid = torch.sigmoid
        #self.save_hyperparameters()

        self.train_f1 = torchmetrics.F1(num_classes=3)
        self.valid_f1 = torchmetrics.F1(num_classes=3)
        self.train_auc = torchmetrics.AUROC(num_classes=3, compute_on_step=False)
        self.valid_auc = torchmetrics.AUROC(num_classes=3, compute_on_step=False)

    def forward(self, x):
        x = torch.squeeze(x, dim=0) 
        features = self.pretrained_model.features(x) # shape s x 256 x 7 x 7
        pooled_features = self.pooling_layer(features) # shape s x 256
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0] # shape 256
        output = self.classifer(flattened_features) # shape 3
        return output

    def cross_entropy_loss(self, logits, labels):
        return F.binary_cross_entropy_with_logits(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        preds = self.sigmoid(logits)
        y_int = y.long()
        self.log("train_f1", self.train_f1(preds, y_int), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("train_AUC", self.train_auc(preds, y_int), on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x) # [[-inf, +inf], [...], [..]]
        loss = self.cross_entropy_loss(logits, y)
        preds = self.sigmoid(logits)
        y_int = y.long()
        self.log("val_f1", self.valid_f1(preds, y_int), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("val_AUC", self.valid_auc(preds, y_int), on_step=False, on_epoch=True)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
