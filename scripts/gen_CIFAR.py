## Standard libraries
import os
import math
import numpy as np

## Imports for plotting
import matplotlib.pyplot as plt

from model import DIYPixelCNNGray
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

from pixelcnn import PixelCNN, SimplePixelCNN, SimplePixelCNNNoStack, trainPixelCNN, sample
import matplotlib.pyplot as plt


# Path to the folder where the datasets are/should be downloaded (e.g. MNIST)
DATASET_PATH = "../data"
# Path to the folder where the pretrained models are saved
SAVE_PATH = "../saved_models"

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)


# 2. Model, optimiser, loss
model = SimplePixelCNN(input_channels=3, hidden_channels=120, kernel_size=5)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_module = nn.CrossEntropyLoss()

# Convert images from 0-1 to 0-255 (integers). We use the long datatype as we will use the images as labels as well
def discretize(sample):
    return (sample * 255).to(torch.long)

test_transform = transforms.Compose([transforms.ToTensor(),
                                     discretize
                                     ])
# For training, we add some augmentation. Networks are too powerful and would overfit.
train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomResizedCrop((32,32), scale=(0.8,1.0), ratio=(0.9,1.1)),
                                      transforms.ToTensor(),
                                      discretize
                                     ])
# Loading the training dataset. We need to split it into a training and validation part
# We need to do a little trick because the validation set should not use the augmentation.
train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=train_transform, download=True)
val_dataset = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)

# We define a set of data loaders that we can use for various purposes later.
train_loader = data.DataLoader(train_dataset, batch_size=40, shuffle=True, drop_last=True, pin_memory=True, num_workers=0)
val_loader = data.DataLoader(val_dataset, batch_size=40, shuffle=False, drop_last=False, pin_memory=True, num_workers=0)

"""
# Convert images from 0-1 to 0-255 (integers). We use the long datatype as we will use the images as labels as well
def discretize(sample):
    return (sample * 255).to(torch.long)

# Transformations applied on each image => only make them a tensor
transform = transforms.Compose([transforms.ToTensor(),
                                discretize])

train_set = MNIST(root=DATASET_PATH, train=True, transform=transform, download=True)

# Loading the test set
val_set = MNIST(root=DATASET_PATH, train=False, transform=transform, download=True)

# We define a set of data loaders that we can use for various purposes later.
train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=False, pin_memory=True, num_workers=0)
val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False, num_workers=0)
"""
"""
batch, _ = next(iter(train_loader))
images = batch[:4].to(device)

show_imgs(torch.cat([images, sample(model, [4, 3, 32, 32], device, mode_name=os.path.join("gen_CIFAR", "v5_gen_CIFAR.tar"), SAVE_PATH=SAVE_PATH, temp=1e-5, img=images.clone())], dim=0))
exit()
"""

#best: 0.75
trainPixelCNN(model=model, loss_module=loss_module, optimizer=optimizer, train_data_loader=train_loader, validation_data_loader=val_loader, 
              device=device, SAVE_PATH=SAVE_PATH, num_epochs=300, model_name="gen_CIFAR", folder_name="gen_CIFAR", load_checkpoint=-1)



#epoch 4: 4.476  4.425