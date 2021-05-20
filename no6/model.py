import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import pytorch_lightning as pl


class Net(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.feature_extractor = resnet18(pretrained=True)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.fc = nn.Linear(1000, 2)
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        
    def forward(self, x):
        h = self.feature_extractor(x)
        h = self.fc(h)
        return h
    
    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        y_label = torch.argmax(y, dim=1)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc(y_label, t), on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        y_label = torch.argmax(y, dim=1)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc(y_label, t), on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        y_label = torch.argmax(y, dim=1)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc(y_label, t), on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer