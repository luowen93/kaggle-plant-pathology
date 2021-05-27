# Contains functions and classes related to data and transformation of data
import os
import cv2
import matplotlib.pyplot as plt
import random
import torchvision.transforms as transforms
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

# Function that returns hard coded data transforms used throughout the process
# This function isn't necessary but makes it easier to make
# sure all code that uses transforms will use the same transform


def define_train_val_transforms():

    train_tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    val_tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    return train_tf, val_tf

# Custom dataset loader to be used with datamodule
class MyDataset(Dataset):

    def __init__(self):
        super().__init__()

    def __getitem__(self, idx):
        # Custom code to obtain data from any structure
        pass
    
    # Code to determine the length of the dataset
    def __len__(self):
        pass

# Data module to be used with the pl.Trainer
class PathologyDatamodule(pl.LightningDataModule):

    def __init__(self, train_ds, val_ds, batch_size = 32):
        super().__init__()
        self.batch_size = batch_size
        self.train_ds = train_ds
        self.val_ds = val_ds

    # Usually we download data, or simply load into a dataset
    def prepare_data(self):
        pass
    
    # Split data sets, depending on what stage (train/test)
    def setup(self, stage = None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size = self.batch_size, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size = self.batch_size, shuffle = False)

    # def test_dataloader(self):
    #     return DataLoader(self.test_ds, batch_size = 32, shuffle = False)

