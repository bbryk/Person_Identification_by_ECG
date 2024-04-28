from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


import torch
import torch.nn as nn


class ResNet1D(nn.Module):
    def __init__(self):
        super(ResNet1D, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.resblocks1 = nn.Sequential(
            ResidualBlock(64, 64, kernel_size=7),
            ResidualBlock(64, 64, kernel_size=7),
            ResidualBlock(64, 128, kernel_size=7),
            ResidualBlock(128, 128, kernel_size=7),
            # ResidualBlock(128, 256, kernel_size=7),
            ResidualBlock(128, 256, kernel_size=7)
        )

        self.fc = nn.Linear(256, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.resblocks1(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
