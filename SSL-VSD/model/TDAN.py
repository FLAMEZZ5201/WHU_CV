import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from .DeepLabV3 import DeepLabV3
from .BDRAR_model import BDRAR
from .resnet_dilation import resnet50, Bottleneck, conv1x1
from collections import OrderedDict

# from networks.Deformable import ConvOffset2d
from Deformable2.deform_conv import DeformConv, _DeformConv, DeformConvPack

from torch.cuda.amp import autocast
import pdb


class Res_Block(nn.Module):
    def __init__(self):
        super(Res_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.batch1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.batch2 = nn.BatchNorm2d(64)

    def forward(self, x):
        res = self.conv1(x)
        res = self.batch1(res)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.batch2(res)
        return self.relu(x + res)


class OurModel(nn.Module):
    def __init__(self):
        super(OurModel, self).__init__()

        self.pre_stage_network = BDRAR()

        self.name = 'TDAN'
        self.conv_first = nn.Sequential(
            nn.Conv2d(4, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.residual_layer = self.make_layer(Res_Block, 5)
        self.relu = nn.ReLU(inplace=True)
        # deformable
        self.cr = nn.Conv2d(128, 64, 3, padding=1, bias=True)
        self.off2d_1 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.off2d_11 = nn.Conv2d(18*8, 18*8, 1, padding=0, bias=True)
        self.dconv_1 = DeformConv(64, 64, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1,
                                   deformable_groups=8, im2col_step=1, bias=False)
        # self.dconv_1 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d_2 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.deconv_2 = DeformConv(64, 64, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1,
                                  deformable_groups=8, im2col_step=1, bias=False)
        # self.deconv_2 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d_3 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.deconv_3 = DeformConv(64, 64, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1,
                                  deformable_groups=8, im2col_step=1, bias=False)
        # self.deconv_3 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.dconv = DeformConv(64, 64, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1,
                                  deformable_groups=8, im2col_step=1, bias=False)
        # self.dconv = ConvOffset2d(64, 64, (3, 3), padding=(1, 1), num_deformable_groups=8)
        # self.recon_lr = nn.Conv2d(64, 3, 3, padding=1, bias=True)
        #
        # fea_ex = [nn.Conv2d(5 * 3, 64, 3, padding= 1, bias=True),
        #                nn.ReLU()]
        #
        # self.fea_ex = nn.Sequential(*fea_ex)
        self.fea_fuse = nn.Sequential(
            nn.Conv2d(192, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.recon_layer = self.make_layer(Res_Block, 10)
        self.pred = nn.Sequential(nn.Conv2d(64, 8, 3, padding=1, stride=1, bias=False),
                                   nn.BatchNorm2d(8),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(0.1),
                                   nn.Conv2d(8, 1, 1)
                                   )
        # upscaling = [
        #     Upsampler(default_conv, 4, 64, act=False),
        #     nn.Conv2d(64, 3, 3, padding=1, bias=False)]
        #
        # self.up = nn.Sequential(*upscaling)

        # xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        self.pre_stage_network.load_state_dict(torch.load('/mnt/data1/czpp/BDRAR-master/ckpt/BDRAR/3001.pth'))

        nn.init.constant_(self.off2d_11.weight, 1 / 2)
        nn.init.constant_(self.off2d_11.bias, 0)

    def align(self, x, x_center):
        y = []
        batch_size, num, ch, w, h = x.size()
        center = num // 2
        ref = x[:, center, :, :, :].clone()
        for i in range(num):
            if i == center:
                y.append(x_center.unsqueeze(1))
                continue
            supp = x[:, i, :, :, :]
            fea = torch.cat([ref, supp], dim=1)
            fea = self.cr(fea)
            # feature trans
            offset1 = self.off2d_1(fea)
            fea = (self.dconv_1(fea, offset1))
            offset2 = self.off2d_2(fea)
            fea = (self.deconv_2(fea, offset2))
            offset3 = self.off2d_3(fea)
            fea = (self.deconv_3(supp, offset3))
            offset4 = self.off2d(fea)
            aligned_fea = (self.dconv(fea, offset4))
            im = self.recon_lr(aligned_fea).unsqueeze(1)
            y.append(im)
        y = torch.cat(y, dim=1)
        return y

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, clip, flow1, flow2):
        with autocast():
            pre_stage_res = [self.pre_stage_network(frame) for frame in clip]

            clip = [torch.cat([pred, frame], dim=1) for frame, pred in zip(clip, pre_stage_res)]

            f0 = self.residual_layer(self.conv_first(clip[0]))
            f1 = self.residual_layer(self.conv_first(clip[1]))
            f2 = self.residual_layer(self.conv_first(clip[2]))

            fea = self.cr(torch.cat([f0, f1], dim=1))
            # feature trans
            offset1 = self.off2d_11(self.off2d_1(fea) + flow1.repeat(1, 72, 1, 1))
            fea = (self.dconv_1(fea.float(), offset1.float()))
            offset2 = self.off2d_2(fea) + offset1
            fea = (self.deconv_2(fea.float(), offset2.float()))
            offset3 = self.off2d_3(fea) + offset2
            fea = (self.deconv_3(f1.float(), offset3.float()))
            offset4 = self.off2d(fea)
            aligned_fea1 = (self.dconv(fea.float(), offset4.float()))

            fea = self.cr(torch.cat([f0, f2], dim=1))
            # feature trans
            offset1 = self.off2d_11(self.off2d_1(fea) + flow2.repeat(1, 72, 1, 1))
            fea = (self.dconv_1(fea.float(), offset1.float()))
            offset2 = self.off2d_2(fea) + offset1
            fea = (self.deconv_2(fea.float(), offset2.float()))
            offset3 = self.off2d_3(fea) + offset2
            fea = (self.deconv_3(f2.float(), offset3.float()))
            offset4 = self.off2d(fea)
            aligned_fea2 = (self.dconv(fea.float(), offset4.float()))

            fuse_fea = torch.cat([f0, aligned_fea1, aligned_fea2], dim=1)
            fea = self.fea_fuse(fuse_fea)
            fea = self.recon_layer(fea)
            pred = self.pred(fea) + pre_stage_res[0]
            
        return pred