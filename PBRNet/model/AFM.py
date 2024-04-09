import torch.nn as nn


class AFM(nn.Module):
    def __init__(self, in_channles, num_channles):
        super(AFM, self).__init__()
        self.conv1 = nn.Conv2d(in_channles, num_channles, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channles, num_channles, kernel_size=1)
        self.sig = nn.Sigmoid()
    def forward(self, x, y):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.sig(x1)
        y1 = self.conv1(y)
        y1 = self.conv2(y1)
        y1 = self.sig(y1)
        out = x1 * x + y1 * y
        return out

