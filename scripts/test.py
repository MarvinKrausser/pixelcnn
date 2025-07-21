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

from pixelcnn import GatedMaskedConv, SimpleGated, SimpleHorizontalStack, SimpleMaskedConvolution, HorizontalStackConvolution, SimplePixelCNN, SimpleVerticalStack, VerticalStackConvolution
import matplotlib.pyplot as plt


torch.set_grad_enabled(False)
torch.set_printoptions(linewidth=200, profile="full")

torch.manual_seed(3)
torch.cuda.manual_seed_all(3)

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

def init_pixelcnn(pixelcnn):
    pixelcnn.conv_init_hor.conv.weight.data.fill_(10)
    pixelcnn.conv_init_hor.conv.bias.data.fill_(0)

    pixelcnn.conv_init_ver.conv.weight.data.fill_(10)
    pixelcnn.conv_init_ver.conv.bias.data.fill_(0)

    pixelcnn.conv_out.conv.weight.data.fill_(10)
    pixelcnn.conv_out.conv.bias.data.fill_(0)

    for layer in pixelcnn.layers:
        layer.conv_hor.conv.weight.data.fill_(10)
        layer.conv_hor.conv.bias.data.fill_(0)

        layer.conv_to_out.conv.weight.data.fill_(10)
        layer.conv_to_out.conv.bias.data.fill_(0)

        layer.conv_ver.conv.weight.data.fill_(10)
        layer.conv_ver.conv.bias.data.fill_(0)

        layer.conv_ver_to_hor.weight.data.fill_(10)
        layer.conv_ver_to_hor.bias.data.fill_(0)

    return pixelcnn

pixelcnn = SimplePixelCNN(input_channels=3, hidden_channels=9, kernel_size=3, dilation_pattern=[1,1,2,2,   1,2,4,4,  1,2,4,4,   1,1,1])
pixelcnn.to(device)
pixelcnn.eval()
pixelcnn.double()
pixelcnn = init_pixelcnn(pixelcnn)


torch.set_grad_enabled(True)
img = torch.full([1, 3, 32, 32], dtype=torch.float32, device=device, fill_value=100)
img.requires_grad_(True)

model = SimplePixelCNN(input_channels=3, hidden_channels=129, kernel_size=3, dilation_pattern=[1,1,2,2,   1,2,4,4,  1,2,4,4,   1,1,1])
model.eval()
model.to(device)

out = model(img)
out = torch.sum(out, dim=1)
loss = out[0, 1, 5, 5]
loss.backward(retain_graph=True)

influence = img.grad[0].clone().detach()
influence = torch.ceil(influence.abs())
for c in range(3):
    print(f"Channel {c}:\n{influence[c]}\n")

img.grad.zero_()

loss = out[0, 2, 31, 31]
loss.backward(retain_graph=True)

influence = img.grad[0].clone().detach()
influence = torch.ceil(influence.abs())
for c in range(3):
    print(f"Channel {c}:\n{influence[c]}\n")

img.grad.zero_()

loss = out[0, 0, 0, 0]
loss.backward(retain_graph=True)

influence = img.grad[0].clone().detach()
influence = torch.ceil(influence.abs())
for c in range(3):
    print(f"Channel {c}:\n{influence[c]}\n")
torch.set_grad_enabled(False)


#pixelcnn check for successor
image_list = []
for c in range(3):
    for h in range(9):
        for w in range(9):
            image = torch.ones((1, 3, 9, 9), dtype=torch.float64, device=device)
            image[:, :, :h, :] = 0
            image[:, :, h, :w] = 0
            image[:, :c, h, w] = 0
            image_list.append(image)

image = torch.cat(image_list, dim=0)

batches = torch.split(image, 20, dim=0)
image_list = [pixelcnn.calculate(batch) for batch in batches]

image = torch.cat(image_list, dim=0)

counter = 0
for c in range(3):
    for h in range(9):
        for w in range(9):
            for cl in range(256):
                if image[counter, cl, c, h, w] != 0:
                    print(cl, c, h, w)
                    print(image[counter].size())
                    print(counter)
                    for c in range(3):
                        print(f"Channel {c}:\n{image[counter, cl][c]}\n")
            counter += 1
if counter != image.size(0):
    raise Exception("Not all images checked", counter, image.size(0))


#pixelcnn check for predecessor
image_list = []
for c in range(3):
    for h in range(9):
        for w in range(9):
            for c_2 in range(3):
                for h_2 in range(h):
                    for w_2 in range(9):
                        image = torch.zeros((1, 3, 9, 9), dtype=torch.float64, device=device)
                        image[:, c_2, h_2, w_2] = 1
                        image_list.append(image)

            for c_2 in range(3):
                for w_2 in range(w):
                    image = torch.zeros((1, 3, 9, 9), dtype=torch.float64, device=device)
                    image[:, c_2, h, w_2] = 1
                    image_list.append(image)
            
            for c_2 in range(c):
                image = torch.zeros((1, 3, 9, 9), dtype=torch.float64, device=device)
                image[:, c_2, h, w] = 1
                image_list.append(image)

image = torch.cat(image_list, dim=0)

batches = torch.split(image, 20, dim=0)

image_list = []
for batch in batches:
    pred = pixelcnn.calculate(batch)
    pred = pred * 0.05
    pred = pred.prod(dim=1)
    image_list.append(pred)

image = torch.cat(image_list, dim=0)

counter = 0
for c in range(3):
    for h in range(9):
        for w in range(9):
            for c_2 in range(3):
                for h_2 in range(h):
                    for w_2 in range(9):
                        if image[counter, c, h, w] == 0:
                            print(f"center: {c}, {h}, {w} | predecessor: {c_2}, {h_2}, {w_2}")
                            for c in range(3):
                                print(f"Channel {c}:\n{image[counter][c]}\n")
                        counter += 1

            for c_2 in range(3):
                for w_2 in range(w):
                    if image[counter, c, h, w] == 0:
                        print(f"center: {c}, {h}, {w} | predecessor: {c_2}, {h}, {w_2}")
                        for c in range(3):
                            print(f"Channel {c}:\n{image[counter][c]}\n")
                    counter += 1
                    
            for c_2 in range(c):
                if image[counter, c, h, w] == 0:
                    print(f"center: {c}, {h}, {w} | predecessor: {c_2}, {h}, {w}")
                    for c in range(3):
                        print(f"Channel {c}:\n{image[counter][c]}\n")
                counter += 1
if counter != image.size(0):
    raise Exception("Not all images checked", counter, image.size(0))




conv = SimpleGated(c_in=3, kernel_size=3)
conv.to(device)
conv.eval()
conv_hor_1 = SimpleHorizontalStack(c_in=3, c_out=3, kernel_size=3, mask_center=True)
conv_hor_1.to(device)
conv_hor_1.eval()

conv_hor_2 = SimpleHorizontalStack(c_in=3, c_out=3, kernel_size=3, mask_center=False)
conv_hor_2.to(device)
conv_hor_2.eval()


conv.conv_hor.conv.weight.fill_(1)
conv.conv_hor.conv.bias.fill_(0)

conv.conv_to_out.conv.weight.fill_(1)
conv.conv_to_out.conv.bias.fill_(0)

conv.conv_ver.conv.weight.fill_(1)
conv.conv_ver.conv.bias.fill_(0)

conv.conv_ver_to_hor.weight.fill_(1)
conv.conv_ver_to_hor.bias.fill_(0)

conv_hor_1.conv.weight.fill_(1)
conv_hor_1.conv.bias.fill_(0)

conv_hor_2.conv.weight.fill_(1)
conv_hor_2.conv.bias.fill_(0)


#horizontal check for successor
image_list = []
for c in range(3):
    for h in range(9):
        for w in range(9):
            image = torch.ones((1, 3, 9, 9), dtype=torch.float32, device=device)
            image[:, :, h, :w] = 0
            image[:, :c, h, w] = 0
            image_list.append(image)

image = torch.cat(image_list, dim=0)

image = conv_hor_1(image)
for _ in range(9):
    image = conv_hor_2(image)

counter = 0
for c in range(3):
    for h in range(9):
        for w in range(9):
            if image[counter, c, h, w] != 0:
                print(image[counter], c, h, w)
            counter += 1
if counter != image.size(0):
    raise Exception("Not all images checked", counter, image.size(0))



#horizontal check for predecessor
image_list = []
for c in range(3):
    for h in range(9):
        for w in range(9):
            for c_2 in range(3):
                for w_2 in range(w):
                    image = torch.zeros((1, 3, 9, 9), dtype=torch.float32, device=device)
                    image[:, c_2, h, w_2] = 1
                    image_list.append(image)

            for c_2 in range(c):
                image = torch.zeros((1, 3, 9, 9), dtype=torch.float32, device=device)
                image[:, c_2, h, w] = 1
                image_list.append(image)

image = torch.cat(image_list, dim=0)

image = conv_hor_1(image)
for _ in range(9):
    image = conv_hor_2(image)

counter = 0
for c in range(3):
    for h in range(9):
        for w in range(9):
            for c_2 in range(3):
                for w_2 in range(w):
                    if image[counter, c, h, w] == 0:
                        print(f"center: {c}, {h}, {w} | predecessor: {c_2}, {h}, {w_2}")
                        for c in range(3):
                            print(f"Channel {c}:\n{image[counter][c]}\n")
                    counter += 1

            for c_2 in range(c):
                if image[counter, c, h, w] == 0:
                    print(f"center: {c}, {h}, {w} | predecessor: {c_2}, {h}, {w}")
                    for c in range(3):
                        print(f"Channel {c}:\n{image[counter][c]}\n")
                counter += 1
if counter != image.size(0):
    raise Exception("Not all images checked", counter, image.size(0))


#gated check for successor
image_list = []
for c in range(3):
    for h in range(9):
        for w in range(9):
            image = torch.ones((1, 3, 9, 9), dtype=torch.float32, device=device)
            image[:, :, :h, :] = 0
            image[:, :, h, :w] = 0
            image[:, :c+1, h, w] = 0
            image_list.append(image)

image = torch.cat(image_list, dim=0)

zero_row = torch.zeros((image.size(0), image.size(1), 1, image.size(3)), dtype=image.dtype, device=image.device)

v_stack_shifted = image[:, :, :-1, :]

v_stack_shifted = torch.cat([zero_row, v_stack_shifted], dim=2)

for _ in range(10):
    image, v_stack_shifted = conv(image, v_stack_shifted)

counter = 0
for c in range(3):
    for h in range(9):
        for w in range(9):
            if image[counter, c, h, w] != 0:
                print(image[counter], c, h, w)
            counter += 1
if counter != image.size(0):
    raise Exception("Not all images checked", counter, image.size(0))



#gated check for predecessor
image_list = []
for c in range(3):
    for h in range(9):
        for w in range(9):
            for c_2 in range(3):
                for h_2 in range(h):
                    for w_2 in range(9):
                        image = torch.zeros((1, 3, 9, 9), dtype=torch.float32, device=device)
                        image[:, c_2, h_2, w_2] = 1
                        image_list.append(image)

            for c_2 in range(3):
                for w_2 in range(w):
                    image = torch.zeros((1, 3, 9, 9), dtype=torch.float32, device=device)
                    image[:, c_2, h, w_2] = 1
                    image_list.append(image)
            
            for c_2 in range(c+1):
                image = torch.zeros((1, 3, 9, 9), dtype=torch.float32, device=device)
                image[:, c_2, h, w] = 1
                image_list.append(image)

image = torch.cat(image_list, dim=0)

zero_row = torch.zeros((image.size(0), image.size(1), 1, image.size(3)), dtype=image.dtype, device=image.device)

v_stack_shifted = image[:, :, :-1, :]

v_stack_shifted = torch.cat([zero_row, v_stack_shifted], dim=2)

for _ in range(10):
    image, v_stack_shifted = conv(image, v_stack_shifted)

counter = 0
for c in range(3):
    for h in range(9):
        for w in range(9):
            for c_2 in range(3):
                for h_2 in range(h):
                    for w_2 in range(9):
                        if image[counter, c, h, w] == 0:
                            print(f"center: {c}, {h}, {w} | predecessor: {c_2}, {h_2}, {w_2}")
                            for c in range(3):
                                print(f"Channel {c}:\n{image[counter][c]}\n")
                        counter += 1

            for c_2 in range(3):
                for w_2 in range(w):
                    if image[counter, c, h, w] == 0:
                        print(f"center: {c}, {h}, {w} | predecessor: {c_2}, {h}, {w_2}")
                        for c in range(3):
                            print(f"Channel {c}:\n{image[counter][c]}\n")
                    counter += 1
                    
            for c_2 in range(c+1):
                if image[counter, c, h, w] == 0:
                    print(f"center: {c}, {h}, {w} | predecessor: {c_2}, {h}, {w}")
                    for c in range(3):
                        print(f"Channel {c}:\n{image[counter][c]}\n")
                counter += 1
if counter != image.size(0):
    raise Exception("Not all images checked", counter, image.size(0))


conv = SimpleMaskedConvolution(c_in=9, c_out=9, kernel_size=3, mask_center=True)
mask = conv.mask

print("SimpleMaskedConvolution Masked")
for output in mask[:3]:
    for input in output[:3]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: r | in: r")

    for input in output[3:6]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: r | in: g")

    for input in output[6:]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: r | in: b")

for output in mask[3:6]:
    for input in output[:3]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: g | in: r")


    for input in output[3:6]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: g | in: g")

    for input in output[6:]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: g | in: b")

for output in mask[6:9]:
    for input in output[:3]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: b | in: r")

    for input in output[3:6]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: b | in: g")

    for input in output[6:]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: b | in: b")

conv = SimpleMaskedConvolution(c_in=9, c_out=9, kernel_size=3, mask_center=False)
mask = conv.mask

print("SimpleMaskedConvolution")
for output in mask[:3]:
    for input in output[:3]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: r | in: r")

    for input in output[3:6]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: r | in: g")

    for input in output[6:]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: r | in: b")

for output in mask[3:6]:
    for input in output[:3]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: g | in: r")


    for input in output[3:6]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: g | in: g")

    for input in output[6:]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: g | in: b")

for output in mask[6:9]:
    for input in output[:3]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: b | in: r")

    for input in output[3:6]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: b | in: g")

    for input in output[6:]:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: b | in: b")

conv = SimpleVerticalStack(c_in=9, c_out=9, kernel_size=3)
mask = conv.mask

print("SimpleVerticalStack")
for output in mask:
    for input in output:
        if not torch.equal(input, torch.tensor([[1, 1, 1], [1, 1, 1], [0, 0, 0]]).float()):
            print(f"{input}: SimpleVerticalStack")

conv = SimpleHorizontalStack(c_in=9, c_out=9, kernel_size=3, mask_center=False)
mask = conv.mask

print("SimpleHorizontalStack")
for output in mask[:3]:
    for input in output[:3]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: r | in: r")

    for input in output[3:6]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: r | in: g")

    for input in output[6:]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: r | in: b")

for output in mask[3:6]:
    for input in output[:3]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: g | in: r")


    for input in output[3:6]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: g | in: g")

    for input in output[6:]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: g | in: b")

for output in mask[6:9]:
    for input in output[:3]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: b | in: r")

    for input in output[3:6]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: b | in: g")

    for input in output[6:]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: b | in: b")

conv = SimpleHorizontalStack(c_in=9, c_out=9, kernel_size=3, mask_center=True)
mask = conv.mask

print("SimpleHorizontalStack Masked")
for output in mask[:3]:
    for input in output[:3]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: r | in: r")

    for input in output[3:6]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: r | in: g")

    for input in output[6:]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: r | in: b")

for output in mask[3:6]:
    for input in output[:3]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: g | in: r")


    for input in output[3:6]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: g | in: g")

    for input in output[6:]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: g | in: b")

for output in mask[6:9]:
    for input in output[:3]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: b | in: r")

    for input in output[3:6]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 1, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: b | in: g")

    for input in output[6:]:
        if not torch.equal(input, torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]).float()):
            print(f"{input}: out: b | in: b")

print("done")