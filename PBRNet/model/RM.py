import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self, in_channles, num_channles, use_1x1conv=False, strides=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channles, num_channles, kernel_size=3, stride=strides, padding=1, )
        self.conv2 = nn.Conv2d(
            num_channles, num_channles, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                in_channles, num_channles, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channles)
        self.bn2 = nn.BatchNorm2d(num_channles)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return F.relu(y)




class RM(nn.Module):
    def __init__(self, in_channles, num_channles):
        super(RM, self).__init__()
        self.conv1 = nn.Conv2d(in_channles, num_channles, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channles, num_channles, kernel_size=1)
        self.res = ResidualBlock(in_channles, num_channles)

    def forward(self, x, y):
        x1 = torch.concat([x, y], dim=-1)
        x1 = self.conv1(x1)
        x1 = self.conv1(x1)
        x1 = self.res(x1)
        x1 = self.res(x1)
        x1 = self.conv1(x1)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        return x1


