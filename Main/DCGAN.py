from __future__ import print_function
import argparse
import os
import random

import torch
from torch import nn, optim, backends
from torch.utils.data import TensorDataset, DataLoader

from torchvision import datasets, transforms
from torchvision.utils import save_image

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
