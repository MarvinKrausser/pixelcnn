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

from model import VerticalStackConvolutionR, HorizontalStackConvolutionR
import matplotlib.pyplot as plt



img = torch.ones(3, 25, 25)
vert_r = VerticalStackConvolutionR(c_in=9, c_out=9, kernel_size=3, mask_center=True)
mask = vert_r.mask

print("out: r")
for image in mask[:3]:
    print("red:")
    for input in image[:3]:
        print(input)


    print("green:")
    for input in image[3:6]:
        print(input)

    print("blue:")
    for input in image[6:]:
        print(input)

print("out: g")
for image in mask[3:6]:
    print("red:")
    for input in image[:3]:
        print(input)


    print("green:")
    for input in image[3:6]:
        print(input)

    print("blue:")
    for input in image[6:]:
        print(input)

print("out: b")
for image in mask[6:9]:
    print("red:")
    for input in image[:3]:
        print(input)


    print("green:")
    for input in image[3:6]:
        print(input)

    print("blue:")
    for input in image[6:]:
        print(input)