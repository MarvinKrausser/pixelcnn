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

from pixelcnn_v2 import PixelCNN, SimpleHorizontalConvolutionGrey, SimpleHorizontalConvolutionRGB, SimpleGatedRGB, SimpleVerticalConvolution
import matplotlib.pyplot as plt

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

torch.set_printoptions(linewidth=200, profile="full")


img = torch.ones(size=(1, 1, 100, 100))
for kernel in range(1, 10):
    for padding in range(10):
        for dilation in range(1, 10):
            for stride in range(1, 10):
                conv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=kernel, padding=padding, dilation=dilation, stride=stride)
                result = conv(img)
                if result.size(3) != (img.size(3) - 1) * stride + dilation * (kernel - 1) - 2 * padding + 1:
                    print(f"result: {result.size(3)}, origin: {img.size(3)}, padding: {padding}, kernel: {kernel}, dilation: {dilation}, calculated: {(img.size(3) - 1) * stride + dilation * (kernel - 1) - 2 * padding}")

exit()


class HelperConstruct(nn.Module):
    def __init__(self, c_in=3, c_hidden=3, kernel_size=3):
        super().__init__()
        self.conv_init_hor = SimpleHorizontalConvolutionRGB(in_channels=c_in, out_channels=c_hidden, kernel_size=kernel_size, mask_center=True)
        self.conv_init_ver = SimpleVerticalConvolution(c_in=c_in, c_out=c_hidden, kernel_size=kernel_size)
        self.conv_out = SimpleHorizontalConvolutionRGB(in_channels=c_in, out_channels=c_hidden, kernel_size=1)
        

    def forward(self, x):
        x = (x.float() / 255.0) * 2 - 1

        zero_row = torch.zeros((x.size(0), x.size(1), 1, x.size(3)), dtype=x.dtype, device=x.device)

        v_stack_shifted = x[:, :, :-1, :]

        v_stack_shifted = torch.cat([zero_row, v_stack_shifted], dim=2)

        x_hor = self.conv_init_hor(x)
        x_ver = self.conv_init_ver(v_stack_shifted)

        x = self.conv_out(x_hor)

        x = self.createOutputRGB(x)

        return x

    def createOutputRGB(self, x):
        r, g, b = x.chunk(3, dim=1)
        r = r.unsqueeze(2)
        g = g.unsqueeze(2)
        b = b.unsqueeze(2)

        # Output dimensions: [Batch, Classes, Channels, Height, Width]
        x = torch.cat([r, g, b], dim=2)

        return x

def resetGrad(model, input):
    if input.grad is not None:
        input.grad.zero_()
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()

def showGrad(model, input, target, dim_to_sum=-1):
    output = model(input)
    output = output.abs()
    if dim_to_sum >= 0:
        output = torch.sum(output, dim=dim_to_sum)
    loss = output[target]
    loss.backward()
    influence = input.grad[0]
    for c in range(input.size(1)):
        print(f"channel: {c}:\n{torch.ceil(influence[c].abs())}\n")
    resetGrad(model, input)


img = torch.zeros(size=[1, 3, 32, 32], dtype=torch.float, requires_grad=True, device=device)

model = PixelCNN(c_in=3, c_hidden=30, kernel_size=5, dilation_pattern=[1,1,2,      1,1])
model.eval()
model.to(device)

print(model(img).size())

showGrad(model, img, (0, 2, 27, 0), dim_to_sum=1)