from __future__ import print_function
import argparse
import os
import random

import torch
from torch import nn, optim, backends
from torch.utils.data import TensorDataset, DataLoader

from torchvision import datasets, transforms
import torchvision.utils as vutils
# from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import import_ipynb

EPOCHS = 500
BATCH_SIZE = 100
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("Current Device : ", DEVICE)

# Set Random Seed for reproductibility
manualSeed = 999
print("Random Seed : ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Store Image-information data and
# Model-Batch-Size information
# at several variables.
dataroot = '../data/'
image_size = 64
batch_size = 128
workers = 2
ngpu = 1  # The Number of available 'gpu'.


"""
    Image-Data Information
    - 1. nc : Number of Input-Image of Color-Channel.
    - 2. nc : Length of Hidden-Vector.
    - 3. ngf : Length of Feature-Map through Generator.
    - 4. gdf : Length of Feature-Map through Discriminator.
"""
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 5
lr = 0.0002
betal = 0.5


train_set = datasets.ImageFolder(
    root=dataroot,
    transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True, num_workers=workers
)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
