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
from itertools import product

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

from pixelcnn_v2 import PixelCNN, SimpleHorizontalConvolutionGrey, SimpleHorizontalConvolutionRGB
import matplotlib.pyplot as plt

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

torch.set_printoptions(linewidth=200, profile="full")

def resetGrad(model, input):
    if input.grad is not None:
        input.grad.zero_()
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()


img = torch.zeros(size=[1, 3, 28, 28], dtype=torch.float, requires_grad=True, device=device)

model = PixelCNN(c_in=3, c_hidden=30, kernel_size=3, dilation_pattern=[1,1,1,      1,1])
model.eval()
model.to(device)

out = model(img)
out = out.abs()
out = torch.sum(out, dim=1)

loss = out[0, 0, 0, 0]
loss.backward()

influence = img.grad[0]
for c in range(img.size(1)):
    print(f"channel: {c}:\n{torch.ceil(influence[c].abs())}\n")
resetGrad(model, img)

exit()

out = model(img)
out = out.abs()
out = torch.sum(out, dim=1)

loss = out[0, 0, 3, 3]
loss.backward()

influence = img.grad[0, 0]
print(f"{torch.ceil(influence.abs())}\n")
resetGrad(model, img)

out = model(img)
out = out.abs()
out = torch.sum(out, dim=1)

loss = out[0, 0, 27, 27]
loss.backward()

influence = img.grad[0, 0]
print(f"{torch.ceil(influence.abs())}\n")
resetGrad(model, img)

out = model(img)
out = out.abs()
out = torch.sum(out, dim=1)

loss = out[0, 0, 27, 0]
loss.backward()

influence = img.grad[0, 0]
print(f"{torch.ceil(influence.abs())}\n")
resetGrad(model, img)