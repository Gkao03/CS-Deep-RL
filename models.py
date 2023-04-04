import torch
import torch.nn as nn
import torch.nn.functional as F
from convgru import ConvGRU


class SharedNet(nn.Module):
    def __init__(self):
        super(SharedNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding='same', dilation=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', dilation=2)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', dilation=3)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', dilation=4)
        self.act4 = nn.ReLU()


    def forward(self, x):
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        out = self.act3(self.conv3(out))
        out = self.act4(self.conv4(out))

        return out
