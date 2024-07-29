from PIL import Image
import os, os.path
import cv2
import imageio
import torch.nn as nn
import glob
import time
import numpy as np
import scipy.ndimage
import math
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.image import imread

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torchvision.io import read_image

df_temp = pd.read_csv('nyu_data/data/nyu2_train.csv', names=['image', 'label'])
df_temp = df_temp.sample(frac = 1 , random_state = 79)
df_train = df_temp[:42000]
df_val = df_temp[42000:]
df_train = df_train.reset_index()
df_train.drop(['index'] , axis = 1 ,inplace = True)

df_val = df_val.reset_index()
df_val.drop(['index'] , axis = 1 ,inplace = True)

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, data_dir, transform=None, target_transform=None):
        '''
        The __init__ function is run once when instantiating the Dataset object
        '''
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        '''
        The __len__ function returns the number of samples in dataset
        '''
        return len(self.data_dir)

    def __getitem__(self, idx):
        '''
        The __getitem__ function loads and returns a sample from the dataset at the given index idx
        '''
        img_path = os.path.join(self.root_dir, self.data_dir['image'][idx])
        image = Image.open(img_path)
        

        label_path = os.path.join(self.root_dir, self.data_dir['label'][idx])
        label = Image.open(label_path)
        
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        
        if self.transform is not None:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            image = self.transform(image)
        if self.target_transform is not None:
            random.seed(seed) # apply this seed to target tranfsorms
            torch.manual_seed(seed) # needed for  torchvision 0.7
            label = self.target_transform(label)
          
        return image, label
        
            
root_dir = './nyu_data'
train_dir = '/kaggle/working/train.csv'
val_dir = '/kaggle/working/val.csv'
transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(transforms=[transforms.ColorJitter(brightness=.3, hue=.1)], p=0.25),
            transforms.RandomApply(transforms=[transforms.GaussianBlur(kernel_size=(11, 11), sigma=(0.1, 5))], p=0.5),
            transforms.Resize((512, 512)), # resize, the smaller edge will be matched.
            transforms.ToTensor(), # convert a PIL image or ndarray to tensor. 
])

val_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((512, 512)), # resize, the smaller edge will be matched.
            transforms.ToTensor(), # convert a PIL image or ndarray to tensor. 
])

target_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((512, 512)), # resize, the smaller edge will be ma
            transforms.ToTensor(), # convert a PIL image or ndarray to tensor. 
])

train_dataset = CustomImageDataset(root_dir, df_train, transform=transform, target_transform=target_transform)
validation_dataset = CustomImageDataset(root_dir, df_val, transform=val_transform, target_transform=target_transform)
train_loader = DataLoader(train_dataset, batch_size=11, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=11, shuffle=True)


import torch
import torch.nn as nn
import torchvision as T

class DSC(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, bias=True):
        super(DSC, self).__init__()
        self.depthwise = nn.Conv2d(
            nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias
        )
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = DSC(in_channels, middle_channels)
        # self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = DSC(middle_channels, out_channels)
        # self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = VGGBlock(ch_out, ch_out, ch_out)

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t), Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            DSC(ch_in, ch_out),
            # nn.Conv2d(ch_in, ch_out, 3, 1, 1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            DSC(F_g, F_int, kernel_size=1, padding=0),
            # nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            DSC(F_l, F_int, kernel_size=1, padding=0),
            # nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            DSC(F_int, 1, kernel_size=1, padding=0),
            # nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class R2AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=32, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=32, ch_out=64, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.Up5 = UpConv(ch_in=512, ch_out=256)
        self.Att5 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_RRCNN5 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up4 = UpConv(ch_in=256, ch_out=128)
        self.Att4 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_RRCNN4 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up3 = UpConv(ch_in=128, ch_out=64)
        self.Att3 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_RRCNN3 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Up2 = UpConv(ch_in=64, ch_out=32)
        self.Att2 = Attention_block(F_g=32, F_l=32, F_int=16)
        self.Up_RRCNN2 = RRCNN_block(ch_in=64, ch_out=32, t=t)

        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

device = "cuda:1"
#weights = R2AttU_Net()
#network = weights.to(device)
weights = R2AttU_Net()
network = weights.to(device)
for name,param in network.named_parameters():
    param.requires_grad = True


import torch.optim as optim
optimizer = optim.Adam(network.parameters(), lr = 0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


min_valid_loss = np.inf
for epoch in range(# Num of epochs):
    total_loss = 0
    #total_correct = 0
    pbar = tqdm(train_loader)
    for batch in pbar:
        if torch.cuda.is_available():
            images,labels = batch
            images, labels = images.to(device), labels.to(device)
        preds = network(images.float().to(device))
        t1 = nn.HuberLoss().to(device)
        #t1 = nn.BCEWithLogitsLoss().to(device)
        loss = t1(preds, labels.float().to(device))
        pbar.set_description(f'Loss -> {loss}')
        pbar.set_postfix_str(f"Lr->{str(scheduler.get_last_lr())}")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.to('cpu').item()
        #total_correct += get_num_correct(preds, labels)
        
    
    valid_loss = 0.0
    network.eval()     # Optional when not using Model Specific layer
    vbar = tqdm(validation_loader)
    scheduler.step()
    with torch.no_grad():
        for vbatch in vbar:
            if torch.cuda.is_available():
                data, labels = vbatch
                data, labels = data.to(device), labels.to(device)

            target = network(data.float()).to(device)
            t1 = nn.HuberLoss().to(device)
            loss = t1(target,labels).to(device)
            valid_loss += loss.to('cpu').item()
            pbar.set_description(f'Validation Loss -> {loss}')

            
    print(f'Epoch {epoch+1} \t\t Training Loss: {total_loss} \t\t Validation Loss: {valid_loss/len(validation_loader)}')
    torch.save(network, f"model_({epoch+1})_{valid_loss/len(validation_loader)}.pth")
        
    print(f'Epoch ==>> {epoch+1}  \t\t Loss ==>> {total_loss/len(train_loader)}')
