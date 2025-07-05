## Standard libraries
import os

## Imports for plotting
import matplotlib.pyplot as plt

plt.set_cmap('cividis')
#%matplotlib inline
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb

## Progress bar
from tqdm import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, padding=padding, dilation=dilation)

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
        mask = torch.ones(c_out, c_in, kernel_size, kernel_size)
        mask[:, :, kernel_size//2+1:,:] = 0

        if(not(c_in % 3 == 0) or not(c_out % 3 == 0)):
            raise Exception("Not divisible by 3")

        # For the very first convolution, we will also mask the center row
        if mask_center:
            mask[                :c_out // 3      ,                :         , kernel_size//2,:] = 0 #Out: r
            mask[c_out // 3      :(c_out * 2) // 3,       c_in // 3:         , kernel_size//2,:] = 0 #Out: g
            mask[(c_out * 2) // 3:                , (c_in * 2) // 3:         , kernel_size//2,:] = 0 #Out: b
        else:
            mask[                :c_out // 3      ,       c_in // 3:         , kernel_size//2,:] = 0 #Out: r
            mask[c_out // 3      :(c_out * 2) // 3, (c_in * 2) // 3:         , kernel_size//2,:] = 0 #Out: g

        super().__init__(c_in, c_out, mask, **kwargs)

class HorizontalStackConvolution(MaskedConvolution):

    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels on the left. Note that our kernel has a size of 1
        # in height because we only look at the pixel in the same row.
        mask = torch.ones(c_out, c_in, kernel_size, kernel_size)
        mask[:, :, :, kernel_size//2+1:] = 0
        mask[:, :, :kernel_size // 2, :] = 0
        mask[:, :, kernel_size // 2 + 1: , :] = 0

        if(not(c_in % 3 == 0) or not(c_out % 3 == 0)):
            raise Exception("Not divisible by 3")

        # For the very first convolution, we will also mask the center pixel
        if mask_center:
            mask[                :c_out // 3      ,                :         , :, kernel_size//2] = 0 #Out: r
            mask[c_out // 3      :(c_out * 2) // 3,       c_in // 3:         , :, kernel_size//2] = 0 #Out: g
            mask[(c_out * 2) // 3:                , (c_in * 2) // 3:         , :, kernel_size//2] = 0 #Out: b
        else:
            mask[                :c_out // 3      ,       c_in // 3:         , :, kernel_size//2] = 0 #Out: r
            mask[c_out // 3      :(c_out * 2) // 3, (c_in * 2) // 3:         , :, kernel_size//2] = 0 #Out: g

        super().__init__(c_in, c_out, mask, **kwargs)

class SimpleMaskedConvolution(MaskedConvolution):
    def __init__(self, c_in, c_out, kernel_size=3, dilation=1, mask_center=False):
        
        mask = torch.ones(c_out, c_in, kernel_size, kernel_size)
        mask[:, :, kernel_size // 2 + 1:, :] = 0
        mask[:, :, kernel_size // 2, kernel_size // 2 + 1:] = 0
        if mask_center:
            mask[                :c_out // 3      ,                :         , kernel_size//2, kernel_size//2] = 0 #Out: r
            mask[c_out // 3      :(c_out * 2) // 3,       c_in // 3:         , kernel_size//2, kernel_size//2] = 0 #Out: g
            mask[(c_out * 2) // 3:                , (c_in * 2) // 3:         , kernel_size//2, kernel_size//2] = 0 #Out: b
        else:
            mask[                :c_out // 3      ,       c_in // 3:         , kernel_size//2, kernel_size//2] = 0 #Out: r
            mask[c_out // 3      :(c_out * 2) // 3, (c_in * 2) // 3:         , kernel_size//2, kernel_size//2] = 0 #Out: g
        super().__init__(c_in=c_in, c_out=c_out, mask=mask, dilation=dilation)

class SimpleVerticalStack(MaskedConvolution):
    def __init__(self, c_in, c_out, kernel_size=3, dilation=1):
        mask = torch.ones(c_out, c_in, kernel_size, kernel_size)
        mask[:, :, kernel_size//2+1:, :] = 0
        super().__init__(c_in=c_in, c_out=c_out, mask=mask, dilation=dilation)

class SimpleHorizontalStack(MaskedConvolution):
    def __init__(self, c_in, c_out, kernel_size=3, dilation=1, mask_center=False):
        mask = torch.ones(c_out, c_in, kernel_size, kernel_size)
        mask[:, :, :kernel_size//2, :] = 0
        mask[:, :, kernel_size//2+1:, :] = 0
        mask[:, :, :, kernel_size//2+1:] = 0

        if mask_center:
            mask[                :c_out // 3      ,                :         , kernel_size//2, kernel_size//2] = 0 #Out: r
            mask[c_out // 3      :(c_out * 2) // 3,       c_in // 3:         , kernel_size//2, kernel_size//2] = 0 #Out: g
            mask[(c_out * 2) // 3:                , (c_in * 2) // 3:         , kernel_size//2, kernel_size//2] = 0 #Out: b
        else:
            mask[                :c_out // 3      ,       c_in // 3:         , kernel_size//2, kernel_size//2] = 0 #Out: r
            mask[c_out // 3      :(c_out * 2) // 3, (c_in * 2) // 3:         , kernel_size//2, kernel_size//2] = 0 #Out: g

        super().__init__(c_in=c_in, c_out=c_out, mask=mask, dilation=dilation)

class SimpleGated(nn.Module):
    def __init__(self, c_in, kernel_size, dilation=1):
        super().__init__()
        self.conv_ver = SimpleVerticalStack(c_in=c_in, c_out=c_in*2, kernel_size=kernel_size, dilation=dilation)
        self.conv_hor = SimpleHorizontalStack(c_in=c_in, c_out=c_in*2, kernel_size=kernel_size, dilation=dilation)
        self.conv_ver_to_hor = nn.Conv2d(in_channels=c_in*2, out_channels=c_in*2, kernel_size=1, padding=0)
        self.conv_to_out = SimpleMaskedConvolution(c_in=c_in, c_out=c_in, kernel_size=1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x_hor, x_ver):
        feat_ver = self.conv_ver(x_ver)
        val, gate = self.split_val_gate_rgb(feat_ver)
        feat_ver_out = torch.tanh(val) * torch.sigmoid(gate)
        feat_ver_out = self.dropout(feat_ver_out)

        feat_hor = self.conv_hor(x_hor)
        feat_hor = feat_hor + self.conv_ver_to_hor(feat_ver)

        val, gate = self.split_val_gate_rgb(feat_hor)
        feat_hor_out = torch.tanh(val) * torch.sigmoid(gate)
        feat_hor_out = self.dropout(feat_hor_out)
        feat_hor_out = self.conv_to_out(feat_hor_out)

        feat_hor_out = x_hor + feat_hor_out
        
        return feat_hor_out, feat_ver_out
    
    def split_val_gate_rgb(self, tensor):
        r, g, b = tensor.chunk(3, dim=1)
        val_r, gate_r = r.chunk(2, dim=1)
        val_g, gate_g = g.chunk(2, dim=1)
        val_b, gate_b = b.chunk(2, dim=1)
        val = torch.cat([val_r, val_g, val_b], dim=1)
        gate = torch.cat([gate_r, gate_g, gate_b], dim=1)
        return val, gate
    
class SimpleGatedNoStack(nn.Module):
    def __init__(self, c_in, kernel_size, dilation=1):
        super().__init__()
        self.conv = SimpleMaskedConvolution(c_in=c_in, c_out=c_in*2, kernel_size=kernel_size, dilation=dilation)
        self.conv_to_out = SimpleMaskedConvolution(c_in=c_in, c_out=c_in, kernel_size=1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        feat = self.conv(x)
        val, gate = self.split_val_gate_rgb(feat)
        feat = torch.tanh(val) * torch.sigmoid(gate)
        feat = self.dropout(feat)
        feat = self.conv_to_out(feat)
        feat = x + feat
        return feat
    
    def split_val_gate_rgb(self, tensor):
        r, g, b = tensor.chunk(3, dim=1)
        val_r, gate_r = r.chunk(2, dim=1)
        val_g, gate_g = g.chunk(2, dim=1)
        val_b, gate_b = b.chunk(2, dim=1)
        val = torch.cat([val_r, val_g, val_b], dim=1)
        gate = torch.cat([gate_r, gate_g, gate_b], dim=1)
        return val, gate
    
class SimplePixelCNNNoStack(nn.Module):
    def __init__(self, input_channels, hidden_channels=66, kernel_size=3):
        super().__init__()


        self.layers = nn.Sequential(
            SimpleMaskedConvolution(c_in=input_channels, c_out=hidden_channels, kernel_size=kernel_size),                              #3
            SimpleGatedNoStack(c_in=hidden_channels, kernel_size=kernel_size),                                                         #5
            SimpleGatedNoStack(c_in=hidden_channels, kernel_size=kernel_size, dilation=2),                                             #9
            SimpleGatedNoStack(c_in=hidden_channels, kernel_size=kernel_size, dilation=4),                                             #17
            SimpleGatedNoStack(c_in=hidden_channels, kernel_size=kernel_size),                                                         #19
            SimpleGatedNoStack(c_in=hidden_channels, kernel_size=kernel_size, dilation=2),                                             #23
            SimpleGatedNoStack(c_in=hidden_channels, kernel_size=kernel_size, dilation=4),                                             #31
            SimpleMaskedConvolution(c_in=hidden_channels, c_out=input_channels*256, kernel_size=1)
        )


    def forward(self, x):
        x = (x.float() / 255.0) * 2 - 1

        x = self.layers(x)

        # Output dimensions: [Batch, Classes, Channels, Height, Width]
        x = x.reshape(x.shape[0], 256, x.shape[1]//256, x.shape[2], x.shape[3])
        return x



class SimplePixelCNN(nn.Module):
    def __init__(self, input_channels, hidden_channels=66, kernel_size=3):
        super().__init__()

        self.conv_init_hor = SimpleHorizontalStack(c_in=input_channels, c_out=hidden_channels, kernel_size=kernel_size, mask_center=True) #2
        self.conv_init_ver = SimpleVerticalStack(c_in=input_channels, c_out=hidden_channels, kernel_size=kernel_size)

        self.layers = nn.ModuleList([
            SimpleGated(c_in=hidden_channels, kernel_size=kernel_size, dilation=1),                                             #4
            SimpleGated(c_in=hidden_channels, kernel_size=kernel_size, dilation=2),                                             #8
            SimpleGated(c_in=hidden_channels, kernel_size=kernel_size, dilation=4),                                             #16
            SimpleGated(c_in=hidden_channels, kernel_size=kernel_size, dilation=1),                                             #18
            SimpleGated(c_in=hidden_channels, kernel_size=kernel_size, dilation=2),                                             #22
            SimpleGated(c_in=hidden_channels, kernel_size=kernel_size, dilation=4)                                              #30
        ])

        self.conv_out = SimpleMaskedConvolution(c_in=hidden_channels, c_out=input_channels*256, kernel_size=1)


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

        # Output dimensions: [Batch, Classes, Channels, Height, Width]
        x = x.reshape(x.shape[0], 256, x.shape[1]//256, x.shape[2], x.shape[3])
        return x




class GatedMaskedConv(nn.Module):

    def __init__(self, c_in, **kwargs):
        """
        Gated Convolution block implemented the computation graph shown above.
        """
        super().__init__()
        self.conv_vert = VerticalStackConvolution(c_in, c_out=2*c_in, mask_center=False, **kwargs)
        self.conv_horiz = HorizontalStackConvolution(c_in, c_out=2*c_in, mask_center=False, **kwargs)
        self.conv_vert_to_horiz = nn.Conv2d(2*c_in, 2*c_in, kernel_size=1, padding=0)
        self.conv_horiz_1x1 = HorizontalStackConvolution(c_in, c_in, kernel_size=1, mask_center=False)
        self.droput = self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, v_stack, h_stack):
        # Vertical stack (left)
        v_stack_feat = self.conv_vert(v_stack)

        v_val, v_gate = self.split_val_gate_rgb(v_stack_feat)
        v_stack_out = torch.tanh(v_val) * torch.sigmoid(v_gate)

        # Create a zero row of same width
        zero_row = torch.zeros((v_stack_feat.size(0), v_stack_feat.size(1), 1, v_stack_feat.size(3)), dtype=v_stack_feat.dtype, device=v_stack_feat.device)

        #Drop last Row
        v_stack_feat = v_stack_feat[:, :, :-1, :]

        #Prepend the zero row
        v_stack_shifted = torch.cat([zero_row, v_stack_feat], dim=2)

        # Horizontal stack (right)
        h_stack_feat = self.conv_horiz(h_stack)

        h_stack_feat = h_stack_feat + self.conv_vert_to_horiz(v_stack_shifted)
        
        h_val, h_gate = self.split_val_gate_rgb(h_stack_feat)

        h_stack_feat = torch.tanh(h_val) * torch.sigmoid(h_gate)
        h_stack_out = self.conv_horiz_1x1(h_stack_feat)
        h_stack_out = h_stack_out + h_stack

        return v_stack_out, h_stack_out
    
    def split_val_gate_rgb(self, tensor):
        r, g, b = tensor.chunk(3, dim=1)
        val_r, gate_r = r.chunk(2, dim=1)
        val_g, gate_g = g.chunk(2, dim=1)
        val_b, gate_b = b.chunk(2, dim=1)
        val = torch.cat([val_r, val_g, val_b], dim=1)
        gate = torch.cat([gate_r, gate_g, gate_b], dim=1)
        return val, gate
    
class PixelCNN(nn.Module):
    def __init__(self, input_channels, hidden_channels=66, kernel_size = 3):
        super().__init__()

        self.act_fn = nn.ELU()
        self.dropout = nn.Dropout2d(p=0.2)

        self.conv_hor_init = HorizontalStackConvolution(c_in=input_channels, c_out=hidden_channels, mask_center=True, kernel_size=kernel_size)
        self.conv_ver_init = VerticalStackConvolution(c_in=input_channels, c_out=hidden_channels, mask_center=False, kernel_size=kernel_size)

        self.masked = nn.ModuleList([
            GatedMaskedConv(c_in=hidden_channels, kernel_size=kernel_size),
            GatedMaskedConv(c_in=hidden_channels, kernel_size=kernel_size, dilation=2),
            GatedMaskedConv(c_in=hidden_channels, kernel_size=kernel_size),
            GatedMaskedConv(c_in=hidden_channels, kernel_size=kernel_size, dilation=4),
            GatedMaskedConv(c_in=hidden_channels, kernel_size=kernel_size),
            GatedMaskedConv(c_in=hidden_channels, kernel_size=kernel_size, dilation=2),
            GatedMaskedConv(c_in=hidden_channels, kernel_size=kernel_size),
            GatedMaskedConv(c_in=hidden_channels, kernel_size=kernel_size, dilation=4),
            GatedMaskedConv(c_in=hidden_channels, kernel_size=kernel_size)
            #GatedMaskedConvR(c_in=hidden_channels, kernel_size=kernel_size, dilation=2),
            #GatedMaskedConvR(c_in=hidden_channels, kernel_size=kernel_size)
        ])
        
        self.conv_out = HorizontalStackConvolution(in_channels=hidden_channels, out_channels=input_channels * 256, kernel_size=1, mask_center=False)

    def forward(self, x):
        x = (x.float() / 255.0) * 2 - 1

        x_hor = self.conv_hor_init(x)
        x_ver = self.conv_ver_init(x)

        for masked in self.masked:
            x_ver, x_hor = masked(x_ver, x_hor)

        out = self.conv_out(self.act_fn(x_hor))

        # Output dimensions: [Batch, Classes, Channels, Height, Width]
        out = out.reshape(out.shape[0], 256, out.shape[1]//256, out.shape[2], out.shape[3])
        return out
    
def sample(model, img_shape, device, SAVE_PATH, mode_name, img=None, temp=1):
    #img_shape(batch, channel, height, width)
    state_dict = torch.load(os.path.join(SAVE_PATH, mode_name), weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    if img == None:
        img = torch.zeros(img_shape).to(device)
    else:
        img = img.to(device)
    for h in tqdm(range(img_shape[2]), desc=f"Generating", leave=False):
        for w in range(img_shape[3]):
            for c in range(img_shape[1]):
                pred = model(img)
                pred = pred * temp
                pred = F.softmax(pred[:,:,c,h,w], dim=-1)
                img[:,c,h,w] = torch.multinomial(pred, num_samples=1).squeeze(dim=-1)
    return img


def trainPixelCNN(model, optimizer, loss_module, train_data_loader, validation_data_loader, device, SAVE_PATH, num_epochs=10, folder_name = "test", model_name="test.tar" , load_checkpoint=-1):
    best_loss = 10
    if load_checkpoint >=0:
        prefix = f"v{load_checkpoint}_" + model_name
        for root, _, files in os.walk(os.path.join(SAVE_PATH, folder_name)):
            for file in files:
                if file.startswith(prefix):
                    rest = file[len(prefix)+1:]
                    best_loss = float(rest)
                    print(best_loss)
                    full_path = os.path.join(root, file)
        state_dict = torch.load(full_path, weights_only=False)
        model.load_state_dict(state_dict)
        load_checkpoint += 1
    else:
        load_checkpoint = 0

    for epoch in range(num_epochs):
        saving = False
        ############
        # Training #
        ############
        total_loss = 0.0  # Track total loss for this epoch
        model.train()
        true_preds, count = 0., 0
        for data_inputs, _ in tqdm(train_data_loader, desc=f"Train Epoch {epoch+1+load_checkpoint}", leave=False):
            data_inputs = data_inputs.to(device)

            preds = model(data_inputs)

            loss = loss_module(preds, data_inputs)
            total_loss += loss.item() * data_inputs.numel()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            true_preds += (preds.argmax(dim=1) == data_inputs).sum().item()
            count += data_inputs.numel()
        train_acc = true_preds / count
        avg_loss_train = total_loss / count

        torch.cuda.empty_cache()

        ##############
        # Validation #
        ##############
        model.eval()

        total_loss = 0.0
        true_preds, count = 0., 0
        for data_inputs, _ in tqdm(validation_data_loader, desc=f"Validate Epoch {epoch+1+load_checkpoint}", leave=False):
            with torch.no_grad():
                data_inputs = data_inputs.to(device)

                preds = model(data_inputs)

                loss = loss_module(preds, data_inputs)
                total_loss += loss.item() * data_inputs.numel()

                true_preds += (preds.argmax(dim=1) == data_inputs).sum().item()
                count += data_inputs.numel()
        val_acc = true_preds / count
        avg_loss_val = total_loss / count

        if(best_loss > avg_loss_val):
            best_loss = avg_loss_val
            saving = True
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, folder_name, f"v{epoch+load_checkpoint}_" + model_name + f"_{best_loss}"))

        print(f"epoch: {epoch+1+load_checkpoint} | train accuracy: {int(train_acc * 1000) / 10}% | train loss: {int(avg_loss_train * 1000) / 1000} | validation loss: {int(avg_loss_val * 1000) / 1000} | validation accuracy: {int(val_acc * 1000) / 10}% | saving: {saving}")
        torch.cuda.empty_cache()