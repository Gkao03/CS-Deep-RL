import torch
import torch.nn as nn
import torch.nn.functional as F
from convgru import ConvGRU


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1, activation=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.activation = activation

    def forward(self, x):
        out = self.conv(x)

        if self.activation is not None:
            out = self.activation(out)

        return out
