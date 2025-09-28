import torch
import torch.nn as nn
import cv2
import os
# from functools import partial
# import torch.nn.functional as F
# import numpy as np
# import torch.optim as optim
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 act_layer=nn.GELU,
                 drop=0.):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CnnDDI(nn.Module):
    def __init__(self):
        super(CnnDDI, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02, inplace=True),

            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=128, kernel_size=3, stride=2, padding=1),  # 128*128*128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),  # 256*32*32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 256*16*16
        )

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.mlp = Mlp(
            in_features=5408,
            hidden_features=int(5408 * 4.),
            out_features= 5408,
            drop = 0.
        )

        self.fc_interaction = nn.Linear(5408, 64)

    def forward(self, img1, img2):
        img1 = self. cnn(img1)
        img1 = img1.reshape(1, 256, -1)
        img1 = img1.reshape(1, 1, 256, -1)
        img2 = self.cnn(img2)
        img2 = img2.reshape(1, 256, -1)
        img2 = img2.reshape(1, 1, 256, -1)
        z = torch.cat((img1, img2), 1)
        z = self.conv2d(z)
        z = z.reshape(1, 256, -1)
        z = self.conv1d(z)
        z = z.reshape(1, -1)
        z = self.mlp(z)
        interaction = self.fc_interaction(z)
        return interaction


model = CnnDDI()
imgpath = 'mol_graph'
drug1_name = 'DB00006'
drug1_name_path = os.path.join(imgpath, str(drug1_name) + '.png')

img1 = cv2.imread('mol_graph/DB00006.png')
img1 = torch.Tensor(img1)
img1 = img1.permute([2, 1, 0])
img1 = img1.unsqueeze(dim=0)
img2 = cv2.imread('mol_graph/DB00014.png')
img2 = torch.Tensor(img2)
img2 = img2.permute([2, 1, 0])
img2 = img2.unsqueeze(dim=0)
output = model(img1, img2)

a = 1

