import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import HP
from augmentations import *


class Encoder(nn.Module):
    def __init__(self, D, device):
        super(Encoder, self).__init__()
        self.resnet = resnet18(pretrained=False).to(device)
        # self.resnet = resnet18(weights=None).to(device)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(512, 512)
        self.fc = nn.Sequential(nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, D))

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

    def encode(self, x):
        return self.forward(x)


class Projector(nn.Module):
    def __init__(self, D, proj_dim):
        super(Projector, self).__init__()
        self.model = nn.Sequential(nn.Linear(D, proj_dim),
                                   nn.BatchNorm1d(proj_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(proj_dim, proj_dim),
                                   nn.BatchNorm1d(proj_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(proj_dim, proj_dim)
                                   )

    def forward(self, x):
        return self.model(x)

    def project(self, x):
        return self.forward(x)


class VICReg(nn.Module):
    def __init__(self, D = HP.ENCODE_D, proj_dim=HP.PROJ_D, device=HP.DEVICE):
        super().__init__()
        self.f = Encoder(D=D, device=device)
        self.h = Projector(D=D, proj_dim=proj_dim)


    def forward(self, X):
        return self.h(self.f(X))

    def encode(self, X):
        return self.f(X)



