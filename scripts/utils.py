## Standard libraries
import os
import math
import numpy as np

## Imports for plotting
import matplotlib.pyplot as plt

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
from torchvision import transforms

def show_tensor_images(tensor, nrow=8, title=None):
    """
    Displays a batch of images from a tensor of shape (B, C, H, W).
    
    Args:
        tensor (torch.Tensor): Image tensor with shape (B, C, H, W)
        nrow (int): Number of images per row in the plot
        title (str): Optional title for the plot
    """
    tensor = tensor.detach().cpu()

    # Normalize if needed (if data is in [0,1] or [0,255])
    if tensor.max() > 1:
        tensor = tensor / 255.0

    batch_size = tensor.size(0)
    ncol = (batch_size + nrow - 1) // nrow

    fig, axs = plt.subplots(ncol, nrow, figsize=(nrow * 2, ncol * 2))

    # If we only have one row, axs is 1D
    axs = axs.flatten() if batch_size > 1 else [axs]

    for i in range(len(axs)):
        axs[i].axis("off")
        if i < batch_size:
            img = tensor[i]
            if img.shape[0] == 1:
                axs[i].imshow(img.squeeze(0), cmap="gray")
            else:
                axs[i].imshow(img.permute(1, 2, 0))  # C, H, W → H, W, C

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def show_imgs(imgs, name=""):
    num_imgs = imgs.shape[0] if isinstance(imgs, torch.Tensor) else len(imgs)
    nrow = min(num_imgs, 4)
    ncol = int(math.ceil(num_imgs/nrow))
    imgs = torchvision.utils.make_grid(imgs, nrow=nrow, pad_value=128)
    imgs = imgs.clamp(min=0, max=255)
    np_imgs = imgs.cpu().numpy()
    plt.figure(figsize=(1.5*nrow, 1.5*ncol))
    plt.imshow(np.transpose(np_imgs, (1,2,0)), interpolation='nearest')
    plt.axis('off')
    plt.title(name)
    plt.show()
    plt.close()


def show_center_recep_field(img, out, title=""):
    """
    Calculates the gradients of the input with respect to the output center pixel,
    and visualizes the overall receptive field.
    Inputs:
        img - Input image for which we want to calculate the receptive field on.
        out - Output features/loss which is used for backpropagation, and should be
              the output of the network/computation graph.
    """
    # Determine gradients
    loss = out[0,:,img.shape[2]//2,img.shape[3]//2].sum() # L1 loss for simplicity
    loss.backward(retain_graph=True) # Retain graph as we want to stack multiple layers and show the receptive field of all of them
    img_grads = img.grad.abs()
    img.grad.fill_(0) # Reset grads

    # Plot receptive field
    img = img_grads.squeeze().cpu().numpy()
    fig, ax = plt.subplots(1,2)
    pos = ax[0].imshow(img)
    ax[1].imshow(img>0)
    # Mark the center pixel in red if it doesn't have any gradients (should be the case for standard autoregressive models)
    show_center = (img[img.shape[0]//2,img.shape[1]//2] == 0)
    if show_center:
        center_pixel = np.zeros(img.shape + (4,))
        center_pixel[center_pixel.shape[0]//2,center_pixel.shape[1]//2,:] = np.array([1.0, 0.0, 0.0, 1.0])
    for i in range(2):
        ax[i].axis('off')
        if show_center:
            ax[i].imshow(center_pixel)
    ax[0].set_title("Weighted receptive field\n" + title)
    ax[1].set_title("Binary receptive field\n" + title)
    plt.show()
    plt.close()