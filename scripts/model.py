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



class SimpleModel(nn.Module):
    def __init__(self, n_input, n_output, n_hidden=[255, 255], act_function=nn.ReLU, init=nn.init.kaiming_normal_):
        super().__init__()
        layers = []
        layers += [nn.Linear(n_input, n_hidden[0]), act_function()]
        for i in range(len(n_hidden)-1):
            layers += [nn.Linear(n_hidden[i], n_hidden[i+1]), act_function()]

        layers += [nn.Linear(n_hidden[len(n_hidden)-1], n_output)]
        self.layers = nn.Sequential(*layers)

        self._initialize_weights(init=init)

    def _initialize_weights(self, init):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
    
class SimpleConv(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, act_function=nn.ReLU, init=nn.init.kaiming_normal_):
        super().__init__()

        self.layers = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 x 16 x 16

            # Conv Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128 x 8 x 8

            # Conv Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 256 x 4 x 4

            nn.Flatten(),

            # Fully connected layers
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

        self._initialize_weights(init=init)

    def _initialize_weights(self, init):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Conv2d):
                init(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.layers(x)
        return x
    
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
        if(mask.dim() == 2):
            kernel_size = (mask.shape[0], mask.shape[1])
        else:
            kernel_size = (mask.shape[2], mask.shape[3])
        dilation = 1 if "dilation" not in kwargs else kwargs["dilation"]
        padding = tuple([dilation*(kernel_size[i]-1)//2 for i in range(2)])
        # Actual convolution
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, padding=padding, **kwargs)

        mask.expand_as(self.conv.weight)

        # Mask as buffer => it is no parameter but still a tensor of the module
        # (must be moved with the devices)
        self.register_buffer('mask', mask)


    def forward(self, x):
        self.conv.weight.data *= self.mask # Ensures zero's at masked positions
        return self.conv(x)
    
class VerticalStackConvolution(MaskedConvolution):

    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels below. For efficiency, we could also reduce the kernel
        # size in height, but for simplicity, we stick with masking here.
        mask = torch.ones(kernel_size, kernel_size)
        mask[kernel_size//2+1:,:] = 0

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

class VerticalStackConvolutionR(MaskedConvolution):

    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels below. For efficiency, we could also reduce the kernel
        # size in height, but for simplicity, we stick with masking here.
        mask = torch.ones(c_out, c_in, kernel_size, kernel_size)
        mask[:, :, kernel_size//2+1:,:] = 0

        if(not(c_in % 3 == 0) or not(c_out % 3 == 0)):
            raise Exception("Not divisible by 3")

        # For the very first convolution, we will also mask the center row
        if mask_center:
            mask[                :c_out // 3      ,          :               , kernel_size//2,:] = 0 #Out: r, in: r,g,b
            mask[c_out // 3      :(c_out * 2) // 3, c_in // 3:               , kernel_size//2,:] = 0 #Out: g, in: g,b
            mask[(c_out * 2) // 3:                , (c_in * 2) // 3:         , kernel_size//2,:] = 0 #Out: b, in: b

        super().__init__(c_in, c_out, mask, **kwargs)

class HorizontalStackConvolutionR(MaskedConvolution):

    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels on the left. Note that our kernel has a size of 1
        # in height because we only look at the pixel in the same row.
        mask = torch.ones(c_out, c_in, 1,kernel_size)
        mask[:, :, 0, kernel_size//2+1:] = 0

        if(not(c_in % 3 == 0) or not(c_out % 3 == 0)):
            raise Exception("Not divisible by 3")

        # For the very first convolution, we will also mask the center pixel
        if mask_center:
            mask[                :c_out // 3      ,          :               , 0, kernel_size//2] = 0 #Out: r, in: r,g,b
            mask[c_out // 3      :(c_out * 2) // 3, c_in // 3:               , 0, kernel_size//2] = 0 #Out: g, in: g,b
            mask[(c_out * 2) // 3:                , (c_in * 2) // 3:         , 0, kernel_size//2] = 0 #Out: b, in: b

        super().__init__(c_in, c_out, mask, **kwargs)

class GatedMaskedConvR(nn.Module):

    def __init__(self, c_in, **kwargs):
        """
        Gated Convolution block implemented the computation graph shown above.
        """
        super().__init__()
        self.conv_vert = VerticalStackConvolutionR(c_in, c_out=2*c_in, mask_center=True, **kwargs)
        self.conv_horiz = HorizontalStackConvolutionR(c_in, c_out=2*c_in, mask_center=True, **kwargs)
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
    

class DIYPixelCNN(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=66):
        super().__init__()

        self.act_fn = nn.ELU()
        self.dropout = nn.Dropout2d(p=0.2)

        self.conv_hor_init = HorizontalStackConvolutionR(c_in=input_channels, c_out=hidden_channels, mask_center=True)
        self.conv_ver_init = VerticalStackConvolutionR(c_in=input_channels, c_out=hidden_channels, mask_center=True)

        self.masked = nn.ModuleList([
            GatedMaskedConvR(c_in=hidden_channels),
            GatedMaskedConvR(c_in=hidden_channels, dilation=2),
            GatedMaskedConvR(c_in=hidden_channels),
            GatedMaskedConvR(c_in=hidden_channels, dilation=4),
            GatedMaskedConvR(c_in=hidden_channels),
            GatedMaskedConvR(c_in=hidden_channels, dilation=8),
            GatedMaskedConvR(c_in=hidden_channels),
            GatedMaskedConvR(c_in=hidden_channels, dilation=16),
            GatedMaskedConvR(c_in=hidden_channels),
            GatedMaskedConvR(c_in=hidden_channels, dilation=32),
            GatedMaskedConvR(c_in=hidden_channels)
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

        return out
    
class DIYPixelCNNGray(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=64):
        super().__init__()

        self.act_fn = nn.ELU()
        self.dropout = nn.Dropout2d(p=0.2)

        self.conv_hor_init = HorizontalStackConvolution(c_in=input_channels, c_out=hidden_channels, mask_center=True)
        self.conv_ver_init = VerticalStackConvolution(c_in=input_channels, c_out=hidden_channels, mask_center=True)

        self.masked = nn.ModuleList([
            GatedMaskedConv(c_in=hidden_channels),
            GatedMaskedConv(c_in=hidden_channels, dilation=2),
            GatedMaskedConv(c_in=hidden_channels),
            GatedMaskedConv(c_in=hidden_channels, dilation=4),
            GatedMaskedConv(c_in=hidden_channels),
            GatedMaskedConv(c_in=hidden_channels, dilation=8),
            GatedMaskedConv(c_in=hidden_channels),
            GatedMaskedConv(c_in=hidden_channels, dilation=16),
            GatedMaskedConv(c_in=hidden_channels),
            GatedMaskedConv(c_in=hidden_channels, dilation=32),
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
    
def sample(model, img_shape, device, SAVE_PATH, mode_name, img=None):
    """
    Sampling function for the autoregressive model.
    Inputs:
        img_shape - Shape of the image to generate (B,C,H,W)
        img (optional) - If given, this tensor will be used as
                        a starting image. The pixels to fill
                        should be -1 in the input tensor.
    """

    state_dict = torch.load(os.path.join(SAVE_PATH, mode_name), weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create empty image
    if img is None:
        img = torch.zeros(img_shape, dtype=torch.long).to(device) - 1
    # Generation loop
    for h in tqdm(range(img_shape[2]), leave=False):
        for w in range(img_shape[3]):
            for c in range(img_shape[1]):
                # Skip if not to be filled (-1)
                if (img[:,c,h,w] != -1).all().item():
                    continue
                # For efficiency, we only have to input the upper part of the image
                # as all other parts will be skipped by the masked convolutions anyways
                pred = model(img[:,:,:h+1,:])
                probs = F.softmax(pred[:,:,c,h,w], dim=-1)
                img[:,c,h,w] = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)
    return img


def trainPixelCNN(model, optimizer, loss_module, train_data_loader, validation_data_loader, test_data_loader, device, SAVE_PATH, num_epochs=10, train=True, folder_name = "test", model_name="test.tar"):

    best_loss = 10

    if(train or not os.path.isfile(os.path.join(SAVE_PATH, model_name))):

        for epoch in range(num_epochs):
            saving = False
            ############
            # Training #
            ############
            total_loss = 0.0  # Track total loss for this epoch
            model.train()
            true_preds, count = 0., 0
            for data_inputs, _ in tqdm(train_data_loader, desc=f"Train Epoch {epoch+1}", leave=False):
                data_inputs = data_inputs.to(device)

                preds = model(data_inputs)

                preds = preds.permute(0, 2, 1, 3, 4).contiguous().view(-1, 256, 32, 32)
                data_inputs = data_inputs.view(-1, 32, 32)

                loss = loss_module(preds, data_inputs)
                total_loss += loss.item()

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                true_preds += (preds.argmax(dim=1) == data_inputs).sum().item()
                count += data_inputs.numel()
            train_acc = true_preds / count
            avg_loss_train = total_loss / len(train_data_loader)

            torch.cuda.empty_cache()

            ##############
            # Validation #
            ##############
            model.eval()

            total_loss = 0.0
            true_preds, count = 0., 0
            for data_inputs, _ in tqdm(validation_data_loader, desc=f"Validate Epoch {epoch+1}", leave=False):
                with torch.no_grad():
                    data_inputs = data_inputs.to(device)

                    preds = model(data_inputs)
                    preds = preds.permute(0, 2, 1, 3, 4).contiguous().view(-1, 256, 32, 32)
                    data_inputs = data_inputs.view(-1, 32, 32)

                    
                    loss = loss_module(preds, data_inputs)
                    total_loss += loss.item()

                    true_preds += (preds.argmax(dim=1) == data_inputs).sum()
                    count += data_inputs.numel()
            val_acc = true_preds / count
            avg_loss_val = total_loss / len(validation_data_loader)

            if(best_loss > avg_loss_val):
                best_loss = avg_loss_val
                saving = True
                torch.save(model.state_dict(), os.path.join(SAVE_PATH, folder_name, f"v{epoch}_" + model_name))

            print(f"epoch: {epoch+1} | train accuracy: {int(train_acc * 1000) / 10}% | train loss: {int(avg_loss_train * 1000) / 1000} | validation loss: {int(avg_loss_val * 1000) / 1000} | validation accuracy: {int(val_acc * 1000) / 10}% | saving: {saving}")
            torch.cuda.empty_cache()


    ###########
    # Testing #
    ###########

    state_dict = torch.load(os.path.join(SAVE_PATH, model_name), weights_only=False)
    model.load_state_dict(state_dict)

    model.eval()

    true_preds, count = 0., 0
    for data_inputs, _ in tqdm(test_data_loader, desc=f"Test", leave=False):
        data_inputs = data_inputs.to(device)

        preds = model(data_inputs)
        preds = preds.permute(0, 2, 1, 3, 4).contiguous().view(-1, 256, 32, 32)
        data_inputs = data_inputs.view(-1, 32, 32)

        true_preds += (preds.argmax(dim=1) == data_inputs).sum()
        count += data_inputs.numel()
    test_acc = true_preds / count
    print(f"test accuracy: {int(test_acc * 1000) / 10}% | best validation loss: {int(best_loss * 1000) / 1000}")

def trainPixelCNNGray(model, optimizer, loss_module, train_data_loader, validation_data_loader, test_data_loader, device, SAVE_PATH, num_epochs=10, train=True, model_name="test.tar", folder_name="test"):

    best_loss = 10

    if(train or not os.path.isfile(os.path.join(SAVE_PATH, model_name))):

        for epoch in range(num_epochs):
            epoch = epoch
            saving = False
            ############
            # Training #
            ############
            total_loss = 0.0  # Track total loss for this epoch
            model.train()
            true_preds, count = 0., 0
            for data_inputs, _ in tqdm(train_data_loader, desc=f"Train Epoch {epoch+1}", leave=False):
                data_inputs = data_inputs.to(device)

                preds = model(data_inputs)

                loss = loss_module(preds, data_inputs)
                total_loss += loss.item()

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                true_preds += (preds.argmax(dim=1) == data_inputs).sum().item()
                count += data_inputs.numel()
            train_acc = true_preds / count
            avg_loss_train = total_loss / len(train_data_loader)

            torch.cuda.empty_cache()

            ##############
            # Validation #
            ##############
            model.eval()

            total_loss = 0.0
            true_preds, count = 0., 0
            for data_inputs, _ in tqdm(validation_data_loader, desc=f"Validate Epoch {epoch+1}", leave=False):
                with torch.no_grad():
                    data_inputs = data_inputs.to(device)

                    preds = model(data_inputs)

                    loss = loss_module(preds, data_inputs)
                    total_loss += loss.item()

                    true_preds += (preds.argmax(dim=1) == data_inputs).sum()
                    count += data_inputs.numel()
            val_acc = true_preds / count
            avg_loss_val = total_loss / len(validation_data_loader)

            if(best_loss > avg_loss_val):
                best_loss = avg_loss_val
                saving = True
                torch.save(model.state_dict(), os.path.join(SAVE_PATH, folder_name, f"v{epoch}_" + model_name))

            print(f"epoch: {epoch+1} | train accuracy: {int(train_acc * 1000) / 10}% | train loss: {int(avg_loss_train * 1000) / 1000} | validation loss: {int(avg_loss_val * 1000) / 1000} | validation accuracy: {int(val_acc * 1000) / 10}% | saving: {saving}")
            torch.cuda.empty_cache()


    ###########
    # Testing #
    ###########

    state_dict = torch.load(os.path.join(SAVE_PATH, model_name), weights_only=False)
    model.load_state_dict(state_dict)

    model.eval()

    true_preds, count = 0., 0
    for data_inputs, _ in tqdm(test_data_loader, desc=f"Test", leave=False):
        data_inputs = data_inputs.to(device)

        preds = model(data_inputs)

        true_preds += (preds.argmax(dim=1) == data_inputs).sum()
        count += data_inputs.numel()
    test_acc = true_preds / count
    print(f"test accuracy: {int(test_acc * 1000) / 10}% | best validation loss: {int(best_loss * 1000) / 1000}")

def train(model, optimizer, loss_module, train_data_loader, validation_data_loader, test_data_loader, device, SAVE_PATH, num_epochs=10, train=True, model_name="test.tar"):

    best_validation = 0

    if(train or not os.path.isfile(os.path.join(SAVE_PATH, "MNSIT_number_save.tar"))):

        for epoch in range(num_epochs):
            ############
            # Training #
            ############
            model.train()
            true_preds, count = 0., 0
            for data_inputs, data_labels in tqdm(train_data_loader, desc=f"Train Epoch {epoch+1}", leave=False):
                data_inputs = data_inputs.to(device)
                data_labels = data_labels.to(device)

                preds = model(data_inputs)
                loss = loss_module(preds, data_labels)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                true_preds += (preds.argmax(dim=-1) == data_labels).sum()
                count += data_labels.shape[0]
            train_acc = true_preds / count


            ##############
            # Validation #
            ##############
            model.eval()

            true_preds, count = 0., 0
            for data_inputs, data_labels in tqdm(validation_data_loader, desc=f"Validate Epoch {epoch+1}", leave=False):
                data_inputs = data_inputs.to(device)
                data_labels = data_labels.to(device)

                preds = model(data_inputs)

                true_preds += (preds.argmax(dim=-1) == data_labels).sum()
                count += data_labels.shape[0]
            val_acc = true_preds / count

            if(best_validation < val_acc):
                best_validation = val_acc
                torch.save(model.state_dict(), os.path.join(SAVE_PATH, model_name))

            print(f"epoch: {epoch+1} | train accuracy: {int(train_acc * 1000) / 10}% | validation accuracy: {int(val_acc * 1000) / 10}%")

    ###########
    # Testing #
    ###########

    wrong_images = []

    state_dict = torch.load(os.path.join(SAVE_PATH, model_name), weights_only=False)
    model.load_state_dict(state_dict)

    model.eval()

    true_preds, count = 0., 0
    for data_inputs, data_labels in tqdm(test_data_loader, desc=f"Test", leave=False):
        data_inputs = data_inputs.to(device)
        data_labels = data_labels.to(device)

        preds = model(data_inputs)

        predicted_labels = preds.argmax(dim=-1)

        # Boolean mask of wrong predictions
        wrong_mask = predicted_labels != data_labels
        
        # Store only the wrongly predicted images
        if wrong_mask.any():
            wrong_images.append(data_inputs[wrong_mask].cpu())

        true_preds += (predicted_labels == data_labels).sum()
        count += data_labels.shape[0]
    test_acc = true_preds / count
    print(f"test accuracy: {int(test_acc * 1000) / 10}% | best validation accuracy: {int(best_validation * 1000) / 10}%")

    show_imgs([wrong_images[i][0] for i in range(20)])



class InceptionBlock(nn.Module):

    def __init__(self, c_in, c_red : dict, c_out : dict, act_fn):
        """
        Inputs:
            c_in - Number of input feature maps from the previous layers
            c_red - Dictionary with keys "3x3" and "5x5" specifying the output of the dimensionality reducing 1x1 convolutions
            c_out - Dictionary with keys "1x1", "3x3", "5x5", and "max"
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super().__init__()

        # 1x1 convolution branch
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(c_in, c_out["1x1"], kernel_size=1),
            nn.BatchNorm2d(c_out["1x1"]),
            act_fn()
        )

        # 3x3 convolution branch
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(c_in, c_red["3x3"], kernel_size=1),
            nn.BatchNorm2d(c_red["3x3"]),
            act_fn(),
            nn.Conv2d(c_red["3x3"], c_out["3x3"], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out["3x3"]),
            act_fn()
        )

        # 5x5 convolution branch
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(c_in, c_red["5x5"], kernel_size=1),
            nn.BatchNorm2d(c_red["5x5"]),
            act_fn(),
            nn.Conv2d(c_red["5x5"], c_out["5x5"], kernel_size=5, padding=2),
            nn.BatchNorm2d(c_out["5x5"]),
            act_fn()
        )

        # Max-pool branch
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(c_in, c_out["max"], kernel_size=1),
            nn.BatchNorm2d(c_out["max"]),
            act_fn()
        )

    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x_max = self.max_pool(x)
        x_out = torch.cat([x_1x1, x_3x3, x_5x5, x_max], dim=1)
        return x_out