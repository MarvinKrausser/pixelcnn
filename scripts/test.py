## Standard libraries
import os
import math
import numpy as np

## Imports for plotting
import matplotlib.pyplot as plt

from utils import show_imgs

plt.set_cmap('cividis')
#%matplotlib inline
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import seaborn as sns

## Progress bar
from tqdm import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms

from pixelcnn import GatedMaskedConv, SimpleHorizontalStack, SimpleMaskedConvolution, HorizontalStackConvolution, SimpleVerticalStack, VerticalStackConvolution
import matplotlib.pyplot as plt

"""
conv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1)
img = torch.ones(1, 4, 4)
img = conv(img)
print(img)
print(img.size())
exit()
"""


conv = SimpleMaskedConvolution(c_in=9, c_out=9, kernel_size=3, mask_center=True)
mask = conv.mask

for output in mask[:3]:
    for input in output[:3]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: r | in: r")

    for input in output[3:6]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: r | in: g")

    for input in output[6:]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: r | in: b")

for output in mask[3:6]:
    for input in output[:3]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: g | in: r")


    for input in output[3:6]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: g | in: g")

    for input in output[6:]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: g | in: b")

for output in mask[6:9]:
    for input in output[:3]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: b | in: r")

    for input in output[3:6]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: b | in: g")

    for input in output[6:]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: b | in: b")

conv = SimpleMaskedConvolution(c_in=9, c_out=9, kernel_size=3, mask_center=False)
mask = conv.mask

for output in mask[:3]:
    for input in output[:3]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: r | in: r")

    for input in output[3:6]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: r | in: g")

    for input in output[6:]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: r | in: b")

for output in mask[3:6]:
    for input in output[:3]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: g | in: r")


    for input in output[3:6]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: g | in: g")

    for input in output[6:]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: g | in: b")

for output in mask[6:9]:
    for input in output[:3]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: b | in: r")

    for input in output[3:6]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: b | in: g")

    for input in output[6:]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: b | in: b")

conv = SimpleVerticalStack(c_in=9, c_out=9, kernel_size=3)
mask = conv.mask

for output in mask:
    for input in output:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 1, 1], [0, 0, 0]]).float()):
            print(f"{input}: SimpleVerticalStack")

conv = SimpleHorizontalStack(c_in=9, c_out=9, kernel_size=3, mask_center=False)
mask = conv.mask

for output in mask[:3]:
    for input in output[:3]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: r | in: r")

    for input in output[3:6]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: r | in: g")

    for input in output[6:]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: r | in: b")

for output in mask[3:6]:
    for input in output[:3]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: g | in: r")


    for input in output[3:6]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: g | in: g")

    for input in output[6:]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: g | in: b")

for output in mask[6:9]:
    for input in output[:3]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: b | in: r")

    for input in output[3:6]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: b | in: g")

    for input in output[6:]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: b | in: b")

conv = SimpleHorizontalStack(c_in=9, c_out=9, kernel_size=3, mask_center=True)
mask = conv.mask

for output in mask[:3]:
    for input in output[:3]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: r | in: r")

    for input in output[3:6]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: r | in: g")

    for input in output[6:]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: r | in: b")

for output in mask[3:6]:
    for input in output[:3]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: g | in: r")


    for input in output[3:6]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: g | in: g")

    for input in output[6:]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: g | in: b")

for output in mask[6:9]:
    for input in output[:3]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: b | in: r")

    for input in output[3:6]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: b | in: g")

    for input in output[6:]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: b | in: b")