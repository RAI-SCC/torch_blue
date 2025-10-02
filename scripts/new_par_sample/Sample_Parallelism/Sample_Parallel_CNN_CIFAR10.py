import torch
from torch import Tensor, nn
import torch_bayesian.vi as vi
from torch_bayesian.vi.variational_distributions import MeanFieldNormalVarDist
from torch.utils.data import DataLoader
from typing import Callable
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import os
import torch.multiprocessing as mp
from torchvision.transforms import ToTensor
from torchvision import datasets
import numpy as np
import random
from Sample_Parallel import seed_all, DDP_pipeline
import sys


class CIFAR10CNN(vi.VIModule):
    def __init__(self, variational_distribution=MeanFieldNormalVarDist()):
        super().__init__()

        # Convolutional Block 1
        self.conv1 = vi.VIConv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1,
                        variational_distribution=variational_distribution)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 2
        self.conv2 = vi.VIConv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1,
                        variational_distribution=variational_distribution)

        # Convolutional Block 3
        self.conv3 = vi.VIConv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1,
                        variational_distribution=variational_distribution)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = vi.VILinear(128 * 4 * 4, 256, variational_distribution=variational_distribution)
        self.fc2 = vi.VILinear(256, 10, variational_distribution=variational_distribution)

    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.conv1(x)))  # (32, 16, 16)

        # Block 2
        x = self.pool(F.relu(self.conv2(x)))  # (64, 8, 8)

        # Block 3
        x = self.pool(F.relu(self.conv3(x)))  # (128, 4, 4)

        # Flatten
        x = self.flatten(x)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    batch_size = 64
    epochs = 10
    random_seed = 42
    global_sample_num = int(sys.argv[1])
    print(f"Global Sample Num: {global_sample_num}")
    print("Sample Parallel MCD on CIFAR10")

    seed = 42
    seed_all(seed, 0)
    
    training_data = datasets.CIFAR10(
    root="/p/scratch/hai_1044/oezdemir1/data",
    train=True,
    download=False,   # IMPORTANT: don't try to download again
    transform=ToTensor())

    test_data = datasets.CIFAR10(
    root="/p/scratch/hai_1044/oezdemir1/data",
    train=False,
    download=False,
    transform=ToTensor())
 
    model = CIFAR10CNN(variational_distribution=MeanFieldNormalVarDist(initial_std=1.))
    model.return_log_probs(False)

    loss_fn = F.cross_entropy
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=0)

    DDP_pipeline(seed, training_data, test_data, model, epochs, batch_size, global_sample_num, optimizer, loss_fn)
    

