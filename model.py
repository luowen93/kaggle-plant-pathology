import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import pytorch_lightning as pl
import pytorch_lightning.metrics as metrics
from optimizers import define_loss


class pathology_classifier(pl.LightningModule):
    def __init__(self, target_classes=10):
        super().__init__()

        self.optimizer = None
        self.scheduler = None
        self.loss = define_loss()

        self.train_accuracy = metrics.Accuracy()
        self.val_accuracy = metrics.Accuracy()

        # Define backbone
        backbone = models.resnet18(pretrained=True)

        # Freeze everything in the backbone
        for child in backbone.children():
            for param in child.parameters():
                param.requires_grad = False

        #         backbone.freeze() # Freeze the model
        numfilters = (
            backbone.fc.in_features
        )  # Number of features in the final average pool
        layers = list(backbone.children())[:-1]  # Take all but the last layer

        self.feature_extractor = nn.Sequential(*layers)
        self.target_classes = target_classes
        self.linear_classifier = nn.Linear(numfilters, 2)

    def forward(self, x):
        self.feature_extractor.eval()  # Change to eval mode

        # No gradients for optimization
        with torch.no_grad():
            features = self.feature_extractor(x).flatten(
                start_dim=1
            )  # Flattens after the first dimension (batch)

        # Calculate the linear weights and perform logsoftmax
        x = self.linear_classifier(features)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.forward(x)
        loss = self.loss(yhat, y)

        self.log("train_loss", loss)
        # Log epoch and accuracy
        self.log("epoch", self.current_epoch)
        self.log("train_acc_step", self.train_accuracy(yhat, y))  # optional metrics

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.forward(x)
        loss = self.loss(yhat, y)
        self.log("validation_loss", loss)
        self.log("epoch", self.current_epoch)
        self.log("val_acc_step", self.val_accuracy(yhat, y))  # optional metrics
        return loss

    # Optional computation of custom metrics on epoch ends
    def training_epoch_end(self, outs):
        self.log("train_acc_epoch", self.train_accuracy.compute())

    def validation_epoch_end(self, outs):
        self.log("val_acc_epoch", self.val_accuracy.compute())

    def configure_optimizers(self):

        if self.optimizer and self.scheduler:
            return self.optimizer, self.scheduler
        elif self.optimizer:
            return self.optimizer
        else:
            print("No optimizer defined")
            return None
