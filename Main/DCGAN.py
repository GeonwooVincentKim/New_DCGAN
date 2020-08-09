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
z_dim = 10
nc = 3
nz = 100
ngf = 64
ndf = 64

num_epochs = 5
lr = 0.0002
beta1 = 0.5
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
    batch_size=IMAGE_SIZE,
    shuffle=True, num_workers=workers
)


"""
    Pytorch represent modules that can handle data easily.    
"""
z = torch.randn(BATCH_SIZE, 64).to(DEVICE)
# real_batch = next(iter(train_loader))
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.title("Training-Images")

# plt.imshow(np.transpose(vutils.make_grid(
#             real_batch[0].to(DEVICE)[:64],
#             padding=2, normalize=True
#         ).cpu(),
#         (1, 2, 0)
#     )
# )
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


"""
    Generator
"""


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input : X Vector
            nn.ConvTranspose2d(
                in_channels=nz,
                out_channels=ngf * 8,
                kernel_size=4,
                stride=1, padding=0,
                bias=False
            ),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # State Size, (ngf * 8) * 4 * 4
            nn.ConvTranspose2d(
                ngf * 8, ngf * 4,
                4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # (ngf * 4) * 8 * 8
            nn.ConvTranspose2d(
                ngf * 4, ngf * 2,
                4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # (ngf * 2) * 16 * 16
            nn.ConvTranspose2d(
                ngf * 2, ngf * 2,
                4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # ngf * 32 * 32
            nn.ConvTranspose2d(
                ngf, nc,
                4, 2, 1, bias=False
            ),
            nn.Tanh()
            # nc * 64 * 64
        ),

    def forward(self, input):
        return self.main


netG = Generator(ngpu).to(DEVICE)

if(DEVICE.type == "cuda") and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(weights_init)
print(netG)


"""
    Discriminator
"""


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input : nc * 64 * 64
            nn.Conv2d(
                nc, ndf,
                4, 2, 1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf * 32 * 32
            nn.Conv2d(
                ndf, ndf * 2,
                4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf * 2) * 16 * 16
            nn.Conv2d(
                ndf * 2, ndf * 4,
                4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf * 4) * 8 * 8
            nn.Conv2d(
                ndf * 4, ndf * 8,
                4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf * 8 ) * 4 * 4
            nn.Conv2d(
                ndf * 8, 1,
                4, 1, 0, bias=False
            ),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


netD = Discriminator(ngpu).to(DEVICE)

if(DEVICE.type == "cuda") and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

netD.apply(weights_init)
print(netD)

"""
    Loss Functions and Optimizers
"""
criterion = nn.BCELoss()
fixed_noise = torch.randn(
     64, nz, 1, 1,
     device=DEVICE
)
real_label = 1
fake_label = 0

optimizerD = list(netD.parameters())
optimizerG = list(netG.parameters())
"""
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.9999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.9999))
"""
