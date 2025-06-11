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
from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST
from torchvision import transforms

class MaskedConvolution(nn.Module):

    def __init__(self, c_in, c_out, mask, **kwargs):
        """
        Implements a convolution with mask applied on its weights.
        Inputs:
            c_in - Number of input channels
            c_out - Number of output channels
            mask - Tensor of shape [kernel_size_H, kernel_size_W] with 0s where
                   the convolution should be masked, and 1s otherwise.
            kwargs - Additional arguments for the convolution
        """
        super().__init__()
        # For simplicity: calculate padding automatically
        kernel_size = (mask.shape[2], mask.shape[3])
        dilation = 1 if "dilation" not in kwargs else kwargs["dilation"]
        padding = tuple([dilation*(kernel_size[i]-1)//2 for i in range(2)])
        # Actual convolution
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, padding=padding, **kwargs)

        # Mask as buffer => it is no parameter but still a tensor of the module
        # (must be moved with the devices)
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.conv.weight.data *= self.mask # Ensures zero's at masked positions
        return self.conv(x)
    
class VerticalStackConvolutionR(MaskedConvolution):

    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels below. For efficiency, we could also reduce the kernel
        # size in height, but for simplicity, we stick with masking here.
        mask = torch.ones(c_out, c_in, kernel_size, kernel_size)
        mask[:, :, kernel_size//2+1:,:] = 0

        # For the very first convolution, we will also mask the center row
        if mask_center:
            mask[0, :, kernel_size//2,:] = 0 #Out: r, in: r,g,b
            mask[1, 1:, kernel_size//2,:] = 0 #Out: g, in: g,b
            mask[2, 2, kernel_size//2,:] = 0 #Out: b, in: b

        super().__init__(c_in, c_out, mask, **kwargs)

class HorizontalStackConvolutionR(MaskedConvolution):

    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels on the left. Note that our kernel has a size of 1
        # in height because we only look at the pixel in the same row.
        mask = torch.ones(c_out, c_in, 1,kernel_size)
        mask[:, :, 0, kernel_size//2+1:] = 0

        # For the very first convolution, we will also mask the center pixel
        if mask_center:
            mask[0, :, 0, kernel_size//2] = 0 #Out: r, in: r,g,b
            mask[1, 1:, 0, kernel_size//2] = 0 #Out: g, in: g,b
            mask[2, 2:, 0, kernel_size//2] = 0 #Out: b, in: b

        super().__init__(c_in, c_out, mask, **kwargs)
    
class VerticalStackConvolution(MaskedConvolution):

    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels below. For efficiency, we could also reduce the kernel
        # size in height, but for simplicity, we stick with masking here.
        mask = torch.ones(c_out, c_in, kernel_size, kernel_size)
        mask[c_out, c_in, kernel_size//2+1:,:] = 0

        # For the very first convolution, we will also mask the center row
        if mask_center:
            mask[kernel_size//2,:] = 0

        super().__init__(c_in, c_out, mask, **kwargs)

class HorizontalStackConvolution(MaskedConvolution):

    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels on the left. Note that our kernel has a size of 1
        # in height because we only look at the pixel in the same row.
        mask = torch.ones(1,kernel_size)
        mask[0,kernel_size//2+1:] = 0

        # For the very first convolution, we will also mask the center pixel
        if mask_center:
            mask[0,kernel_size//2] = 0

        super().__init__(c_in, c_out, mask, **kwargs)


class GatedMaskedConv(nn.Module):

    def __init__(self, c_in, **kwargs):
        """
        Gated Convolution block implemented the computation graph shown above.
        """
        super().__init__()
        self.conv_vert = VerticalStackConvolution(c_in, c_out=2*c_in, **kwargs)
        self.conv_horiz = HorizontalStackConvolution(c_in, c_out=2*c_in, **kwargs)
        self.conv_vert_to_horiz = nn.Conv2d(2*c_in, 2*c_in, kernel_size=1, padding=0)
        self.conv_horiz_1x1 = nn.Conv2d(c_in, c_in, kernel_size=1, padding=0)

    def forward(self, v_stack, h_stack):
        # Vertical stack (left)
        v_stack_feat = self.conv_vert(v_stack)
        v_val, v_gate = v_stack_feat.chunk(2, dim=1)
        v_stack_out = torch.tanh(v_val) * torch.sigmoid(v_gate)

        # Horizontal stack (right)
        h_stack_feat = self.conv_horiz(h_stack)
        h_stack_feat = h_stack_feat + self.conv_vert_to_horiz(v_stack_feat)
        h_val, h_gate = h_stack_feat.chunk(2, dim=1)
        h_stack_feat = torch.tanh(h_val) * torch.sigmoid(h_gate)
        h_stack_out = self.conv_horiz_1x1(h_stack_feat)
        h_stack_out = h_stack_out + h_stack

        return v_stack_out, h_stack_out
    
class PixelCnn(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=64):
        super().__init__()

        self.act_fn = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.2)

        self.conv_hor_init = HorizontalStackConvolution(c_in=input_channels, c_out=hidden_channels, mask_center=True)
        self.conv_ver_init = VerticalStackConvolution(c_in=input_channels, c_out=hidden_channels, mask_center=True)

        self.masked = nn.ModuleList([
            GatedMaskedConv(c_in=hidden_channels),
            GatedMaskedConv(c_in=hidden_channels, dilation=2),
            GatedMaskedConv(c_in=hidden_channels),
            GatedMaskedConv(c_in=hidden_channels, dilation=4),
            GatedMaskedConv(c_in=hidden_channels)
        ])

        self.conv_out = nn.Conv2d(in_channels=hidden_channels, out_channels=input_channels * 256, kernel_size=1, padding=0)

    def forward(self, x):
        x = (x.float() / 255.0) * 2 - 1

        x_hor = self.conv_hor_init(x)
        x_ver = self.conv_ver_init(x)

        for masked in self.masked:
            x_ver, x_hor = masked(x_ver, x_hor)
            x_hor = self.dropout(x_hor)
            x_ver = self.dropout(x_ver)

        out = self.conv_out(self.act_fn(x_hor))

        # Output dimensions: [Batch, Classes, Channels, Height, Width]
        out = out.reshape(out.shape[0], 256, out.shape[1]//256, out.shape[2], out.shape[3])
        return out