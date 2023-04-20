
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualCNN(nn.Module):
    def __init__(self, gray=True):
        super(ResidualCNN, self).__init__()
        input_nc = 1
        if not gray:
            input_nc = 3
        self.conv1 = nn.Conv2d(input_nc, 64, kernel_size=3, padding='same', dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding='same', dilation=2, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding='same', dilation=3, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding='same', dilation=4, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding='same', dilation=3, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding='same', dilation=2, bias=False)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, input_nc, kernel_size=3, padding='same', dilation=1, bias=False)

        self.device = torch.device('cpu')

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.conv7(x)
        return x

    def move(self, device):
        newself = self.to(device)
        newself.device = device
        return newself

class ResidualCNN_noBN(nn.Module):
    def __init__(self, gray=True):
        super(ResidualCNN_noBN, self).__init__()
        input_nc = 1
        if not gray:
            input_nc = 3
        self.conv1 = nn.Conv2d(input_nc, 64, kernel_size=3, padding='same', dilation=1, bias=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding='same', dilation=2, bias=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding='same', dilation=3, bias=True)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding='same', dilation=4, bias=True)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding='same', dilation=3, bias=True)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding='same', dilation=2, bias=True)
        self.conv7 = nn.Conv2d(64, input_nc, kernel_size=3, padding='same', dilation=1, bias=True)

        self.device = torch.device('cpu')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.conv7(x)
        return x

    def move(self, device):
        newself = self.to(device)
        newself.device = device
        return newself
