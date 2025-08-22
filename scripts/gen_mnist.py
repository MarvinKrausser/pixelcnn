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

from pixelcnn_v2 import PixelCNN, trainPixelCNN, sample
import matplotlib.pyplot as plt


# Path to the folder where the datasets are/should be downloaded (e.g. MNIST)
DATASET_PATH = "../data"
# Path to the folder where the pretrained models are saved
SAVE_PATH = "../saved_models"

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)


# 2. Model, optimiser, loss
model = PixelCNN(c_in=1, c_hidden=64, kernel_size=5, dilation_pattern=[1,1,2,      1,1], classes_condition=10)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_module = nn.CrossEntropyLoss(reduction='none')

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

#batch, _ = next(iter(train_loader))
#batch = batch[:12]
#batch[:, :, 20:, :] = -1

condition = torch.tensor((0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
condition = condition.repeat(4)
show_imgs(sample(model, [len(condition), 1, 28, 28], device, model_name="v43_gen_mnist", folder="gen_mnist_condition", SAVE_PATH=SAVE_PATH, temp=1, img=None, condition=condition))
exit()

trainPixelCNN(model=model, loss_module=loss_module, optimizer=optimizer, train_data_loader=train_loader, validation_data_loader=val_loader, 
              device=device, SAVE_PATH=SAVE_PATH, num_epochs=300, model_name="gen_mnist", folder_name="gen_mnist_condition", load_checkpoint=-1, condition=True)
