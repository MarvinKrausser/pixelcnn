## Standard libraries
import os
import math
import numpy as np

## Imports for plotting
import matplotlib.pyplot as plt

from utils import show_imgs, show_tensor_images

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
from torchvision.datasets import CIFAR10, MNIST
from torchvision import transforms

from pixelcnn_v2 import PixelCNN, trainPixelCNN, sample
import matplotlib.pyplot as plt


# Path to the folder where the datasets are/should be downloaded
DATASET_PATH = "../data"
# Path to the folder where the pretrained models are saved
SAVE_PATH = "../saved_models"

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)


# 2. Model, optimiser, loss
model = PixelCNN(c_in=1, c_hidden=90, kernel_size=5, dilation_pattern=[1,1,2,      1,1])
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_module = nn.CrossEntropyLoss(reduction='none')


def discretize(sample):
    return (sample * 255).to(torch.long)

test_transform = transforms.Compose([transforms.Grayscale(1),
                                     transforms.ToTensor(),
                                     discretize
                                     ])

train_transform = transforms.Compose([transforms.Grayscale(1), 
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomResizedCrop((32,32), scale=(0.8,1.0), ratio=(0.9,1.1)),
                                      transforms.ToTensor(),
                                      discretize
                                     ])

train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=train_transform, download=True)
val_dataset = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)


train_loader = data.DataLoader(train_dataset, batch_size=30, shuffle=True, drop_last=True, pin_memory=True, num_workers=0)
val_loader = data.DataLoader(val_dataset, batch_size=30, shuffle=False, drop_last=False, pin_memory=True, num_workers=0)

#batch, _ = next(iter(train_loader))
#batch = batch[:12]
#batch[:, :, 20:, :] = -1

#sample
#show_imgs(sample(model, [2, 3, 32, 32], device, model_name="v7_gen_CIFAR", folder="gen_CIFAR", SAVE_PATH=SAVE_PATH, temp=1, img=None))
#exit()

#train
trainPixelCNN(model=model, loss_module=loss_module, optimizer=optimizer, train_data_loader=train_loader, validation_data_loader=val_loader, 
              device=device, SAVE_PATH=SAVE_PATH, num_epochs=300, model_name="gen_CIFAR", folder_name="gen_CIFAR_gray", load_checkpoint=-1)