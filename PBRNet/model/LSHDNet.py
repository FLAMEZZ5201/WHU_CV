import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from SCNN import SCNN
class LSHDNet(nn.Module):
    def __init__(self, block, layers,input_size):
        super(LSHDNet, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Bottom-up layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # CM
        self.CM1 = CM(input_size=input_size,channel=64)
        self.CM2 = CM(input_size=input_size,channel=128)
        self.CM3 = CM(input_size=input_size, channel=256)
        self.CM4 = CM(input_size=input_size, channel=512)


        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != block.expansion * planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, block.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(block.expansion * planes)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        c2 = self.CM1(c2)
        c3 = self.CM1(c3)
        c4 = self.CM1(c4)
        c5 = self.CM1(c5)

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth

        p2 = self.smooth3(p2)
        return p2

class CM(nn.Module):
    def __init__(self, input_size, channel):
        super(CM, self).__init__()
        self.SCNN1 = SCNN(input_size, ms_ks=1)
        self.SCNN2 = SCNN(input_size, ms_ks=2)
        self.SCNN3 = SCNN(input_size, ms_ks=3)
        self.SCNN4 = SCNN(input_size, ms_ks=4)
        self.deconv = torch.nn.ConvTranspose2d(out_channels=channel)
        self.GP = torch.nn.AdaptiveAvgPool2d(channel)
        self.conv1 = torch.nn.Conv2d(kernel_size=1, in_channels=channel,out_channels=channel/4)
        self.conv2 = torch.nn.Conv2d(kernel_size=1, in_channels=channel,out_channels=channel)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        # SE Branch
        x1 = self.GP(x)
        x1 = self.conv2(self.conv1(x1))
        x1 = self.sig(x1)
        # SCNN Branch
        y1 = self.SCNN1(x)
        y2 = self.SCNN2(x)
        y3 = self.SCNN3(x)
        y4 = self.SCNN4(x)
        y = torch.cat([y1, y2, y3, y4], dim=-1)
        y = self.deconv(y)
        # Residual Branch
        z = self.conv2(x)
        out = x1 * y + z
        return out




