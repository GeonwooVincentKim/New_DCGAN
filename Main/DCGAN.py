from __future__ import print_function
import argparse
import os
import random
from multiprocessing.dummy import freeze_support

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
IMAGE_SIZE = 64   # Set IMAGE_SIZE as 64
BATCH_SIZE = 128  # Originally, it was 100.
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
workers = 2


"""
    Image-Data Information
    - 1. nc : Number of Input-Image of Color-Channel.
    - 2. nc : Length of Hidden-Vector.
    - 3. ngf : Length of Feature-Map through Generator.
    - 4. gdf : Length of Feature-Map through Discriminator.
    
    - 5. num_epochs : Number of training-epoch.
    - 6. Learning-Rate that applied in Training-Model.
    - 7. betal : Hyper-Parameter for Adam-optimizer.
    - 8. ngpu : Number of available gpu.
"""
nc = 3
nz = 100
ngf = 64
ndf = 64

num_epochs = 5
lr = 0.0002
betal = 0.5
ngpu = 1


train_set = datasets.ImageFolder(
    root=dataroot,
    transform=transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=BATCH_SIZE,
    shuffle=True, num_workers=workers
)


"""
    Pytorch represent modules that can handle data easily.    
"""
z = torch.randn(BATCH_SIZE, 64).to(DEVICE)
real_batch = next(iter(train_loader))
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.title("Training-Images")

plt.imshow(np.transpose(vutils.make_grid(
            real_batch[0].to(DEVICE)[:64],
            padding=2, normalize=True
        ).cpu(),
        (1, 2, 0)
    )
)
plt.show()


"""
    Initialize Weights
"""


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# def run():
#     torch.multiprocessing.freeze_support()
#     print("loop")
#
#
# if __name__ == "__main__":
#     freeze_support()
