import math
import os
import sys
import matplotlib.pyplot as plt

from utils import show_imgs

plt.set_cmap('cividis')
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf')
from matplotlib.colors import to_rgb

## Progress bar
from tqdm import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MaskedConvolution(nn.Module):
    def __init__(self, c_in, c_out, mask, dilation = 1):
        super().__init__()
        if mask.dim() != 4:
            raise Exception("Mask not enough Dimensions")
        
        kernel_size = [mask.shape[2], mask.shape[3]]

        padding = tuple([dilation*(kernel_size[i]-1)//2 for i in range(2)])
        self.conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, dilation=dilation, padding=padding, kernel_size=kernel_size)

        self.register_buffer('mask', mask)

    def forward(self, x):
        self.conv.weight.data *= self.mask.to(self.conv.weight.device)
        return self.conv(x)
    

class SimpleMaskedConvolutionRGB(MaskedConvolution):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        if(in_channels % 3 != 0):
            raise Exception("Channel not divisible by 3")
        
        mask = torch.ones(out_channels, in_channels, 1, 1)

        section = out_channels//3
        mask[:section, section:, 0, 0] = 0
        mask[section:section*2, section*2:, 0, 0] = 0

        super().__init__(c_in=in_channels, c_out=out_channels, mask=mask, dilation=1)

class SimpleVerticalConvolution(nn.Module):
    def __init__(self, c_in, c_out, kernel_size = 3, dilation = 1):
        if(kernel_size % 2 == 0):
            raise Exception("Even kernel_size")
        super().__init__()
        kernel_size_hor = kernel_size + self.half_even(kernel_size)

        self.padding_ver = ((kernel_size // 2) * dilation) * 2
        self.padding_hor = dilation * (kernel_size_hor // 2)

        self.conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(kernel_size, kernel_size_hor), dilation=dilation, padding=0)

    def forward(self, x):
        x = F.pad(x, (self.padding_hor, self.padding_hor, self.padding_ver, 0))
        x = self.conv(x)
        return x
    
    def half_even(self, n):
        h = (n + 1) // 2
        return h + (h % 2)

class SimpleHorizontalConvolutionGrey(nn.Module):
    def __init__(self, c_in, c_out, kernel_size = 3, dilation = 1, mask_center=False):
        if(kernel_size % 2 == 0):
            raise Exception("Even kernel_size")
        super().__init__()

        self.padding = ((kernel_size // 2) * dilation) * 2
        if mask_center:
            self.padding += 1
        self.mask_center = mask_center
        self.conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, kernel_size), dilation=dilation, padding=0)

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        if self.mask_center:
            x = x[:, :, :, :-1]
        x = self.conv(x)
        return x
    
class SimpleHorizontalConvolutionRGB(nn.Module):
    def __init__(self, c_in, c_out, kernel_size = 3, dilation = 1, mask_center=False):
        if(kernel_size % 2 == 0):
            raise Exception("Even kernel_size")
        if(c_in % 3 != 0):
            raise Exception("Channel not divisible by 3")
        super().__init__()

        self.padding = ((kernel_size // 2) * dilation) * 2
        self.conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, kernel_size), dilation=dilation, padding=0)

        mask = torch.ones(c_out, c_in, 1, kernel_size)

        if mask_center:
            mask[                :c_out // 3      ,                :         , 0, -1] = 0 #Out: r
            mask[c_out // 3      :(c_out * 2) // 3,       c_in // 3:         , 0, -1] = 0 #Out: g
            mask[(c_out * 2) // 3:                , (c_in * 2) // 3:         , 0, -1] = 0 #Out: b
        else:
            mask[                :c_out // 3      ,       c_in // 3:         , 0, -1] = 0 #Out: r
            mask[c_out // 3      :(c_out * 2) // 3, (c_in * 2) // 3:         , 0, -1] = 0 #Out: g

        self.register_buffer('mask', mask)

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        self.conv.weight.data *= self.mask
        x = self.conv(x)
        return x

class SimpleGatedGrey(nn.Module):
    def __init__(self, c_in, kernel_size=3, dilation=1):
        super().__init__()
        self.conv_ver = SimpleVerticalConvolution(c_in=c_in, c_out=c_in*2, kernel_size=kernel_size, dilation=dilation)
        self.conv_hor = SimpleHorizontalConvolutionGrey(c_in=c_in, c_out=c_in*2, kernel_size=kernel_size, dilation=dilation)
        self.conv_ver_to_hor = nn.Conv2d(in_channels=c_in*2, out_channels=c_in*2, kernel_size=1)
        self.conv_hor_out = nn.Conv2d(in_channels=c_in, out_channels=c_in, kernel_size=1)

    def forward(self, hor, ver):
        feat_ver = self.conv_ver(ver)
        signal_ver, gate_ver = feat_ver.chunk(2, dim=1)
        out_ver = torch.tanh(signal_ver) * torch.sigmoid(gate_ver)

        feat_hor = self.conv_hor(hor)
        feat_hor =  feat_hor + self.conv_ver_to_hor(feat_ver)
        signal_hor, gate_hor = feat_hor.chunk(2, dim=1)
        feat_hor = torch.tanh(signal_hor) * torch.sigmoid(gate_hor)
        feat_hor = self.conv_hor_out(feat_hor)
        hor = hor + feat_hor
        return hor, out_ver
    
class SimpleGatedRGB(nn.Module):
    def __init__(self, c_in, kernel_size=3, dilation=1):
        super().__init__()
        self.conv_ver = SimpleVerticalConvolution(c_in=c_in, c_out=c_in*2, kernel_size=kernel_size, dilation=dilation)
        self.conv_hor = SimpleHorizontalConvolutionRGB(c_in=c_in, c_out=c_in*2, kernel_size=kernel_size, dilation=dilation)
        self.conv_ver_to_hor = nn.Conv2d(in_channels=c_in*2, out_channels=c_in*2, kernel_size=1)
        self.conv_hor_out = SimpleMaskedConvolutionRGB(in_channels=c_in, out_channels=c_in)

    def forward(self, hor, ver):
        feat_ver = self.conv_ver(ver)
        signal_ver, gate_ver = self.split_val_gate_rgb(feat_ver)
        out_ver = torch.tanh(signal_ver) * torch.sigmoid(gate_ver)

        feat_hor = self.conv_hor(hor)
        feat_hor =  feat_hor + self.conv_ver_to_hor(feat_ver)
        signal_hor, gate_hor = self.split_val_gate_rgb(feat_hor)
        feat_hor = torch.tanh(signal_hor) * torch.sigmoid(gate_hor)
        feat_hor = self.conv_hor_out(feat_hor)
        hor = hor + feat_hor
        return hor, out_ver
    
    def split_val_gate_rgb(self, tensor):
        r, g, b = tensor.chunk(3, dim=1)
        val_r, gate_r = r.chunk(2, dim=1)
        val_g, gate_g = g.chunk(2, dim=1)
        val_b, gate_b = b.chunk(2, dim=1)
        val = torch.cat([val_r, val_g, val_b], dim=1)
        gate = torch.cat([gate_r, gate_g, gate_b], dim=1)
        return val, gate

class PixelCNN(nn.Module):
    def __init__(self, c_in, c_hidden, dilation_pattern, kernel_size=3):
        super().__init__()
        if c_in == 1:
            horizontal = SimpleHorizontalConvolutionGrey
            gated = SimpleGatedGrey
            out = nn.Conv2d
            self.outputFnct = self.createOutputGrey
        else:
            horizontal = SimpleHorizontalConvolutionRGB
            gated = SimpleGatedRGB
            out = SimpleMaskedConvolutionRGB
            self.outputFnct = self.createOutputRGB

        self.conv_init_hor = horizontal(c_in=c_in, c_out=c_hidden, kernel_size=kernel_size, mask_center=True)
        self.conv_init_ver = SimpleVerticalConvolution(c_in=c_in, c_out=c_hidden, kernel_size=kernel_size)

        self.layers = nn.ModuleList()

        for pattern in dilation_pattern:
            self.layers.append(gated(c_in=c_hidden, kernel_size=kernel_size, dilation=pattern))

        self.conv_out = out(in_channels=c_hidden, out_channels=256*c_in, kernel_size=1)

    def forward(self, x):
        x = (x.float() / 255.0) * 2 - 1

        zero_row = torch.zeros((x.size(0), x.size(1), 1, x.size(3)), dtype=x.dtype, device=x.device)

        v_stack_shifted = x[:, :, :-1, :]

        v_stack_shifted = torch.cat([zero_row, v_stack_shifted], dim=2)

        x_hor = self.conv_init_hor(x)
        x_ver = self.conv_init_ver(v_stack_shifted)

        for layer in self.layers:
            x_hor, x_ver = layer(x_hor, x_ver)
        
        x = self.conv_out(x_hor)

        x = self.outputFnct(x)

        return x
    
    def createOutputGrey(self, x):
        x = x.reshape(x.shape[0], 256, 1, x.shape[2], x.shape[3])
        return x

    def createOutputRGB(self, x):
        r, g, b = x.chunk(3, dim=1)
        r = r.unsqueeze(2)
        g = g.unsqueeze(2)
        b = b.unsqueeze(2)

        # Output dimensions: [Batch, Classes, Channels, Height, Width]
        x = torch.cat([r, g, b], dim=2)

        return x
    

def trainPixelCNN(model, optimizer, loss_module, train_data_loader, validation_data_loader, device, SAVE_PATH, num_epochs=10, folder_name = "test", model_name="test.tar" , load_checkpoint=-1):
    load_checkpoint, best_loss = loadCheckpoint(model, SAVE_PATH, folder_name, model_name, load_checkpoint)

    for epoch in range(num_epochs):
        saving = False
        ############
        # Training #
        ############
        model.train()
        true_preds, count = 0., 0
        loss_bits_per_dim_train = 0
        for data_inputs, _ in tqdm(train_data_loader, desc=f"Train Epoch {epoch+1+load_checkpoint}", leave=False):
            data_inputs = data_inputs.to(device)

            preds = model(data_inputs)

            loss = loss_module(preds, data_inputs)
            loss_bits_per_dim_train = loss.mean(dim=[1,2,3]) * np.log2(np.exp(1))
            loss_bits_per_dim_train = loss_bits_per_dim_train.mean()

            optimizer.zero_grad()

            loss_bits_per_dim_train.backward()

            optimizer.step()

            true_preds += (preds.argmax(dim=1) == data_inputs).sum().item()
            count += data_inputs.numel()
        train_acc = true_preds / count

        torch.cuda.empty_cache()

        ##############
        # Validation #
        ##############
        model.eval()

        loss_bits_per_dim_valid = 0
        true_preds, count = 0., 0
        for data_inputs, _ in tqdm(validation_data_loader, desc=f"Validate Epoch {epoch+1+load_checkpoint}", leave=False):
            with torch.no_grad():
                data_inputs = data_inputs.to(device)

                preds = model(data_inputs)

                loss = loss_module(preds, data_inputs)
                loss_bits_per_dim_valid = loss.mean(dim=[1,2,3]) * np.log2(np.exp(1))
                loss_bits_per_dim_valid = loss_bits_per_dim_valid.mean()

                true_preds += (preds.argmax(dim=1) == data_inputs).sum().item()
                count += data_inputs.numel()
        val_acc = true_preds / count

        if(best_loss > loss_bits_per_dim_valid):
            best_loss = loss_bits_per_dim_valid
            saving = True
            save_dir = os.path.join(SAVE_PATH, folder_name)
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f"v{epoch+load_checkpoint}_" + model_name + f"_{best_loss}")
            torch.save(model.state_dict(), save_path)

        print(f"epoch: {epoch+1+load_checkpoint} | train accuracy: {int(train_acc * 1000) / 10}% | train loss: {loss_bits_per_dim_train} | validation loss: {loss_bits_per_dim_valid} | validation accuracy: {int(val_acc * 1000) / 10}% | saving: {saving}")
        torch.cuda.empty_cache()


def sample(model, img_shape, device, SAVE_PATH, model_name, folder, img=None, temp=1):
    #img_shape(batch, channel, height, width)
    with torch.no_grad():
        for root, _, files in os.walk(os.path.join(SAVE_PATH, folder)):
            for file in files:
                if file.startswith(model_name):
                    print(f"Filename: {file}")
                    full_path = os.path.join(root, file)
        state_dict = torch.load(full_path, weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        if img == None:
            img = torch.zeros(img_shape, device=device, dtype=torch.float32)-1
        else:
            img = img.to(device)
        for h in tqdm(range(img_shape[2]), desc=f"Generating", leave=False):
            for w in range(img_shape[3]):
                for c in range(img_shape[1]):
                    if (img[:,c,h,w] != -1).all().item():
                            continue
                    pred = model(img[:,:,:h+1,:])
                    probs = F.softmax(pred[:,:,c,h,w] / temp, dim=-1)
                    img[:,c,h,w] = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)
        return img

def loadCheckpoint(model, SAVE_PATH, folder_name, model_name, load_checkpoint):
    best_loss = sys.float_info.max
    if load_checkpoint >=0:
        prefix = f"v{load_checkpoint}_" + model_name
        for root, _, files in os.walk(os.path.join(SAVE_PATH, folder_name)):
            for file in files:
                if file.startswith(prefix):
                    rest = file[len(prefix)+1:]
                    best_loss = float(rest)
                    print(f"Checkpoint loss: {best_loss}")
                    full_path = os.path.join(root, file)
        state_dict = torch.load(full_path, weights_only=False)
        model.load_state_dict(state_dict)
        load_checkpoint += 1
    else:
        load_checkpoint = 0
    return load_checkpoint,best_loss