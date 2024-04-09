import torch.nn as nn
import torch
import torch.nn.functional as F
from .DeepLabV3 import DeepLabV3
from .BDRAR_model import BDRAR
from .resnet_dilation import resnet50, Bottleneck, conv1x1
from collections import OrderedDict
from .resnext_modify import ResNeXt101

# from networks.Deformable import ConvOffset2d
from Deformable2.deform_conv import DeformConv, _DeformConv, DeformConvPack

from torch.cuda.amp import autocast
import pdb

class _ConvBatchNormReLU(nn.Sequential):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        relu=True,
    ):
        super(_ConvBatchNormReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
        )
        self.add_module(
            "bn",
            nn.BatchNorm2d(out_channels),
        )

        if relu:
            self.add_module("relu", nn.ReLU())

    def forward(self, x):
        return super(_ConvBatchNormReLU, self).forward(x)


class _ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling with image pool"""

    def __init__(self, in_channels, out_channels, output_stride):
        super(_ASPPModule, self).__init__()
        if output_stride == 8:
            pyramids = [12, 24, 36]
        elif output_stride == 16:
            pyramids = [6, 12, 18]
        self.stages = nn.Module()
        self.stages.add_module(
            "c0", _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1)
        )
        for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBatchNormReLU(in_channels, out_channels, 3, 1, padding, dilation),
            )
        self.imagepool = nn.Sequential(
            OrderedDict(
                [
                    ("pool", nn.AdaptiveAvgPool2d((1,1))),
                    ("conv", _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1)),
                ]
            )
        )
        self.fire = nn.Sequential(
            OrderedDict(
                [
                    ("conv", _ConvBatchNormReLU(out_channels * 5, out_channels, 3, 1, 1, 1)),
                    ("dropout", nn.Dropout2d(0.1))
                ]
            )
        )

    def forward(self, x):
        h = self.imagepool(x)
        h = [F.interpolate(h, size=x.shape[2:], mode="bilinear", align_corners=False)]
        for stage in self.stages.children():
            h += [stage(x)]
        h = torch.cat(h, dim=1)
        h = self.fire(h)
        return h


class UpsampleConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None, norm=None, bias=True):
        super(UpsampleConvLayer, self).__init__()

        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=upsample, mode='nearest')

        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)

        if self.norm in ["BN" or "IN"]:
            out = self.norm_layer(out)
        out = self.relu(out)
        return out


class Skip_Connect(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Skip_Connect, self).__init__()

        self.bottlneck = _ConvBatchNormReLU(in_channels, out_channels, 3, 1, 1, 1)

    def forward(self, f_last):
        f_last = self.bottlneck(f_last)

        return f_last


class OurModel(nn.Module):
    def __init__(self, pretrained_path=None, num_classes=1, all_channel=256, all_dim=26 * 26,
                 T=0.07):  # 473./8=60 416./8=52
        super(OurModel, self).__init__()

        self.pre_stage_network = BDRAR()

        self.pre_stage_network.load_state_dict(torch.load('/mnt/data1/czpp/BDRAR-master/ckpt/BDRAR/3001.pth'))
        # for p in self.pre_stage_network.parameters():
        #     p.requires_grad = False

        # self.pre_stage_network.load_state_dict(torch.load("/mnt/data1/czpp/FSDNet/ckpt/FSDNet/50000_sbu.pth",
        #                                                   map_location=lambda storage, loc: storage.cuda(0)))

        ## feature extractor

        resnext = ResNeXt101()
        # self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.first_conv = _ConvBatchNormReLU(in_channels=4, out_channels=64, kernel_size=7, stride=2, padding=3, dilation=1)
        # self.feature_extractor = resnet50(pretrained=True, output_stride=16, input_channels=3)
        # self.aspp4 = _ASPPModule(2048, 32, 16)
        # self.aspp3 = _ASPPModule(1024, 32, 16)
        # self.aspp2 = _ASPPModule(512, 32, 16)
        # self.aspp1 = _ASPPModule(256, 32, 16)
        self.aspp4 = nn.Sequential(nn.Conv2d(2048, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.aspp3 = nn.Sequential(nn.Conv2d(1024, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.aspp2 = nn.Sequential(nn.Conv2d(512, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.aspp1 = nn.Sequential(nn.Conv2d(256, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))

        self.pred = nn.Sequential(nn.Conv2d(32, 8, 3, padding=1, stride=1, bias=False),
                                   nn.BatchNorm2d(8),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(0.1),
                                   nn.Conv2d(8, 1, 1)
                                   )

        # deformabel 4
        self.off2d_4 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.deconv_4 = DeformConv(64, 64, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1,
                                   deformable_groups=8, im2col_step=1, bias=False)
        # self.norm4 = nn.BatchNorm2d(64)

        # deformable 2
        self.off2d_21 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.off2d_22 = nn.Conv2d(18 * 8, 18 * 8, 1, padding=0, bias=True)
        self.deconv_2 = DeformConv(32, 32, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1,
                                   deformable_groups=8, im2col_step=1, bias=False)
        # self.norm2 = nn.BatchNorm2d(32)
        nn.init.constant_(self.off2d_22.weight, 1 / 2)
        nn.init.constant_(self.off2d_22.bias, 0)

        # deformable 1
        self.off2d_11 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.off2d_12 = nn.Conv2d(18 * 8, 18 * 8, 1, padding=0, bias=True)
        self.deconv_1 = DeformConv(32, 32, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1,
                                   deformable_groups=8, im2col_step=1, bias=False)
        # self.norm1 = nn.BatchNorm2d(32)
        nn.init.constant_(self.off2d_12.weight, 1 / 2)
        nn.init.constant_(self.off2d_12.bias, 0)
        # # self.deconv_1 = ConvOffset2d(256, 256, 3, padding=1, num_deformable_groups=8)
        # self.deconv_1 = DeformConv(64, 64, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, deformable_groups=8, im2col_step=1, bias=False)
        # self.norm1 = nn.BatchNorm2d(64)
        # # self.conv11 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        # self.off2d_2 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        # # self.deconv_2 = ConvOffset2d(256, 256, 3, padding=1, num_deformable_groups=8)
        # self.deconv_2 = DeformConv(64, 64, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, deformable_groups=8, im2col_step=1, bias=False)
        #
        # self.norm2 = nn.BatchNorm2d(64)
        # self.off2d_3 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        # # self.deconv_2 = ConvOffset2d(256, 256, 3, padding=1, num_deformable_groups=8)
        # self.deconv_3 = DeformConv(64, 64, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1,
        #                            deformable_groups=8, im2col_step=1, bias=False)
        # self.norm3 = nn.BatchNorm2d(64)
        # self.off2d_3 = nn.Conv2d(256, 18 * 8, 3, padding=1, bias=True)
        # self.deconv_3 = ConvOffset2d(256, 256, 3, padding=1, num_deformable_groups=8)
        # self.deconv_3 = DeformConv(128, 128, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, deformable_groups=8, im2col_step=1, bias=False)
        # self.off2d_4 = nn.Conv2d(128, 18 * 8, 3, padding=1, bias=True)
        # self.deconv_4 = ConvOffset2d(256, 256, (3, 3), padding=(1, 1), num_deformable_groups=8)
        # self.deconv_4 = DeformConv(128, 128, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, deformable_groups=8, im2col_step=1, bias=False)

        # decoder
        # res 4         ref1 + ref2 + f4 + f3
        self.refine4 = nn.Sequential(
            nn.Conv2d(64 + 64 + 32 + 32, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        # upsample
        self.deconv4 = UpsampleConvLayer(32, 32, kernel_size=3, stride=1, upsample=2, bias=False,
                                         norm='BN')

        # res 3         ref1 + ref2 + f2 + f3
        self.refine3 = nn.Sequential(
            nn.Conv2d(32 + 32 + 32 + 32, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        # upsample
        self.deconv3 = UpsampleConvLayer(32, 32, kernel_size=3, stride=1, upsample=2, bias=False,
                                         norm='BN')

        # res 2         ref1 + ref2 + f2 + f1
        self.refine2 = nn.Sequential(
            nn.Conv2d(32 + 32 + 32 + 32, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        # upasmple
        self.deconv2 = UpsampleConvLayer(32, 32, kernel_size=3, stride=1, upsample=2, bias=False,
                                         norm='BN')
        # upasmple
        self.deconv1 = UpsampleConvLayer(32, 32, kernel_size=3, stride=1, upsample=2, bias=False,
                                         norm='BN')
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialize_weights(self.pred, self.deconv1, self.deconv2, self.deconv3, self.deconv4, self.refine4, self.refine3, self.refine2,
        #                    self.aspp1, self.aspp2, self.aspp3, self.aspp4)

        # self.fusion_predict = nn.Conv2d(1, 1, 1, bias=False)
        # nn.init.constant_(self.fusion_predict.weight, 1 / 2)

    def feature_extract(self, x):
        x = self.first_conv(x)                      # 1/2   64
        f1 = self.layer1(x)                         # 1/4   256
        f2 = self.layer2(f1)                        # 1/8   512
        f3 = self.layer3(f2)                        # 1/16  1024
        f4 = self.layer4(f3)                        # 1/16  2048

        f4 = self.aspp4(f4)                          # 1/16  32
        f3 = self.aspp3(f3)                          # 1/16  32
        f2 = self.aspp2(f2)                          # 1/8  32
        f1 = self.aspp1(f1)                          # 1/4  32

        return f1, f2, f3, f4

    def forward(self, clip, flow1_ref1, flow1_ref2, flow2_ref1, flow2_ref2):
        with autocast():
            pre_stage_res = [self.pre_stage_network(frame) for frame in clip]

            clip = [torch.cat([pred, frame], dim=1) for frame, pred in zip(clip, pre_stage_res)]

            # feature extractor
            f1, f2, f3, f4 = self.feature_extract(clip[0])
            f1_ref1, f2_ref1, _, f4_ref1 = self.feature_extract(clip[1])
            f1_ref2, f2_ref2, _, f4_ref2 = self.feature_extract(clip[2])

            # f_ref1 = self.first_conv(clip[1])
            # f_ref1_1 = self.layer1(f_ref1)
            # f_ref1_2 = self.layer2(f_ref1_1)
            # f_ref1 = self.layer3(f_ref1_2)
            # f_ref1_4 = self.layer4(f_ref1)
            # f_ref1 = self.aspp4(f_ref1_4)
            #
            # f_ref2 = self.first_conv(clip[2])
            # f_ref2 = self.layer1(f_ref2)
            # f_ref2 = self.layer2(f_ref2)
            # f_ref2 = self.layer3(f_ref2)
            # f_ref2 = self.layer4(f_ref2)
            # f_ref2 = self.aspp4(f_ref2)
            # pdb.set_trace()

            # pred_example = self.pred1(F.upsample(f4, size=clip[1].size()[2:], mode='bilinear'))

            # with autocast(enabled=False):
            f_t = torch.cat((f4, f4_ref1), dim=1)
            offset = self.off2d_4(f_t)
            # f4_ref1 = self.relu(self.norm4(self.deconv_4(f_t.float(), offset.float())))
            f4_ref1 = self.deconv_4(f_t.float(), offset.float())

            f_t = torch.cat((f4, f4_ref2), dim=1)
            offset = self.off2d_4(f_t)
            # f4_ref2 = self.relu(self.norm4(self.deconv_4(f_t.float(), offset.float())))
            f4_ref2 = self.deconv_4(f_t.float(), offset.float())

            f_cur_t = torch.cat((f4_ref1, f4_ref2, f4, f3), dim=1)
            f_cur = self.relu(self.refine4(f_cur_t) + f4)                                         # f3_predict
            # pred_res3 = self.pred(F.upsample(f_cur, size=clip[1].size()[2:], mode='bilinear'))
            f_cur = self.deconv4(f_cur)

            f_t = torch.cat((f2, f2_ref1), dim=1)
            offset = self.off2d_22(self.off2d_21(f_t) + flow1_ref1.repeat(1, 72, 1, 1))
            # f2_ref1 = self.relu(self.norm2(self.deconv_2(f2_ref1.float(), offset.float())))
            f2_ref1 = self.deconv_2(f2_ref1.float(), offset.float())

            f_t = torch.cat((f2, f2_ref2), dim=1)
            offset = self.off2d_22(self.off2d_21(f_t) + flow1_ref2.repeat(1, 72, 1, 1))
            # f2_ref2 = self.relu(self.norm2(self.deconv_2(f2_ref2.float(), offset.float())))
            f2_ref2 = self.deconv_2(f2_ref2.float(), offset.float())

            f_cur_t = torch.cat((f2_ref1, f2_ref2, f_cur, f2), dim=1)
            f_cur = self.relu(self.refine3(f_cur_t) + f_cur)                                         # f2_predict
            # pred_res2 = self.pred(F.upsample(f_cur, size=clip[1].size()[2:], mode='bilinear'))
            f_cur = self.deconv3(f_cur)

            f_t = torch.cat((f1, f1_ref1), dim=1)
            offset = self.off2d_12(self.off2d_11(f_t) + flow2_ref1.repeat(1, 72, 1, 1))
            # f1_ref1 = self.relu(self.norm1(self.deconv_1(f1_ref1.float(), offset.float())))
            f1_ref1 = self.deconv_1(f1_ref1.float(), offset.float())

            f_t = torch.cat((f1, f1_ref2), dim=1)
            offset = self.off2d_12(self.off2d_11(f_t) + flow2_ref2.repeat(1, 72, 1, 1))
            # f1_ref2 = self.relu(self.norm1(self.deconv_1(f1_ref2.float(), offset.float())))
            f1_ref2 = self.deconv_1(f1_ref2.float(), offset.float())

            f_cur_t = torch.cat((f1_ref1, f1_ref2, f_cur, f1), dim=1)
            f_cur = self.relu(self.refine2(f_cur_t) + f_cur)                                        # f1_predict
            # pred_res1 = self.pred(F.upsample(f_cur, size=clip[1].size()[2:], mode='bilinear'))
            f_cur = self.deconv2(f_cur)
            f_cur = self.deconv1(f_cur)

            # res = self.fusion_predict(self.pred(f_cur) + pre_stage_res[0])
            res = self.pred(f_cur) + pre_stage_res[0]

            # f_ref1_t = torch.cat((f4, f_ref1_t), dim=1)
            # offset2 = self.off2d_2(f_ref1_t) + offset1
            # f_ref1_t = self.relu(self.norm2(self.deconv_2(f_ref1.float(), offset2.float())))
            # 
            # f_ref1_t = torch.cat((f4, f_ref1_t), dim=1)
            # offset3 = self.off2d_3(f_ref1_t) + offset2
            # f_ref1 = self.relu(self.norm3(self.deconv_3(f_ref1.float(), offset3.float())))
            # 
            # # f_ref1_t = torch.cat((f4, f_ref1), dim=1)
            # # offset1 = self.off2d_11(self.off2d_1(f_ref1_t) + flow_ref1_example.repeat(1, 72, 1, 1))
            # # f_ref1_t = self.relu(self.norm1(self.deconv_1(f_ref1_t.float(), offset1.float())))
            # # 
            # # # f_ref1_t = torch.cat((f4, f_ref1), dim=1)
            # # offset2 = self.off2d_2(f_ref1_t)
            # # f_ref1_t = self.relu(self.norm2(self.deconv_2(f_ref1_t.float(), offset2.float())))
            # # 
            # # # f_ref1_t = torch.cat((f4, f_ref1), dim=1)
            # # offset3 = self.off2d_3(f_ref1_t)
            # # f_ref1 = self.relu(self.norm3(self.deconv_3(f_ref1_t.float(), offset3.float())))
            # 
            # # f_ref1 = self.relu(self.conv11(f_ref1))
            # # offset2 = self.off2d_2(f_ref1)
            # # f_ref1 = self.deconv_2(f_ref1.float(), offset2.float())
            # # offset3 = self.off2d_3(f_ref1)
            # # f_ref1 = self.deconv_3(f_ref1.float(), offset3.float())
            # # offset4 = self.off2d_4(f_ref1)
            # # f_ref1 = self.deconv_4(f_ref1.float(), offset4.float())
            # 
            # # f_ref2_t = torch.cat((f4, f_ref2), dim=1)
            # # offset1 = self.off2d_11(self.off2d_1(f_ref2_t) + flow_ref2_example.repeat(1, 72, 1, 1))
            # # f_ref2_t = self.relu(self.norm1(self.deconv_1(f_ref2.float(), offset1.float())))
            # # 
            # # f_ref2_t = torch.cat((f4, f_ref2_t), dim=1)
            # # offset2 = self.off2d_2(f_ref2_t) + offset1
            # # f_ref2_t = self.relu(self.norm2(self.deconv_2(f_ref2.float(), offset2.float())))
            # # 
            # # f_ref2_t = torch.cat((f4, f_ref2_t), dim=1)
            # # offset3 = self.off2d_3(f_ref2_t) + offset2
            # # f_ref2 = self.relu(self.norm3(self.deconv_3(f_ref2.float(), offset3.float())))
            # 
            # f_ref2_t = torch.cat((f4, f_ref2), dim=1)
            # offset1 = self.off2d_11(self.off2d_1(f_ref2_t) + flow_ref2_example.repeat(1, 72, 1, 1))
            # f_ref2_t = self.relu(self.norm1(self.deconv_1(f_ref2_t.float(), offset1.float())))
            # 
            # # f_ref2_t = torch.cat((f4, f_ref2), dim=1)
            # offset2 = self.off2d_2(f_ref2_t)
            # f_ref2_t = self.relu(self.norm2(self.deconv_2(f_ref2_t.float(), offset2.float())))
            # 
            # # f_ref2_t = torch.cat((f4, f_ref2), dim=1)
            # offset3 = self.off2d_3(f_ref2_t)
            # f_ref2 = self.relu(self.norm3(self.deconv_3(f_ref2_t.float(), offset3.float())))
            # 
            # # f_ref2 = self.relu(self.conv11(f_ref2))
            # # offset2 = self.off2d_2(f_ref2)
            # # f_ref2 = self.deconv_2(f_ref2.float(), offset2.float())
            # # offset3 = self.off2d_3(f_ref2)
            # # f_ref2 = self.deconv_3(f_ref2.float(), offset3.float())
            # # offset4 = self.off2d_4(f_ref2)
            # # f_ref2 = self.deconv_4(f_ref2.float(), offset4.float())
            # 
            # ## fuse
            # f_cur = self.relu(self.refine1(torch.cat((f4, f_ref1, f_ref2, f3), dim=1)) + f4)
            # f_cur = self.relu(self.deconv1(f_cur))
            # f_cur = self.relu(self.refine2(torch.cat((f_cur, f2), dim=1)) + f_cur)
            # f_cur = self.relu(self.deconv2(f_cur))
            # f_cur = self.relu(self.refine3(torch.cat((f_cur, f1), dim=1)) + f_cur)
            # f_cur = self.relu(self.deconv3(f_cur))
            # f_cur = self.relu(self.deconv4(f_cur))
            # # f_cur = self.relu(self.deconv4(f_cur))
            # 
            # res = self.fusion_predict(self.pred1(f_cur) + pre_stage_res[0])

            # return res, pred_res3, pred_res2, pred_res1
            return res

    # def forward(self, clip, flow_ref1_example, flow_ref2_example, flow_ref1_example2, flow_ref2_example2):
    #     with autocast():
    #         pre_stage_res = [self.pre_stage_network(frame) for frame in clip]
    #
    #         clip = [torch.cat([pred, frame], dim=1) for frame, pred in zip(clip, pre_stage_res)]
    #
    #         ## feature extractor
    #         f1, f2, f3, f4 = self.feature_extract(clip[0])
    #
    #         f_ref1 = self.first_conv(clip[1])
    #         f_ref1 = self.feature_extractor.maxpool(f_ref1)
    #         f_ref1 = self.feature_extractor.layer1(f_ref1)
    #         f_ref1_2 = self.feature_extractor.layer2(f_ref1)
    #         f_ref1 = self.feature_extractor.layer3(f_ref1_2)
    #         f_ref1 = self.feature_extractor.layer4(f_ref1)
    #         f_ref1 = self.aspp(f_ref1)
    #
    #         f_ref2 = self.first_conv(clip[2])
    #         f_ref2 = self.feature_extractor.maxpool(f_ref2)
    #         f_ref2 = self.feature_extractor.layer1(f_ref2)
    #         f_ref2_2 = self.feature_extractor.layer2(f_ref2)
    #         f_ref2 = self.feature_extractor.layer3(f_ref2_2)
    #         f_ref2 = self.feature_extractor.layer4(f_ref2)
    #         f_ref2 = self.aspp(f_ref2)
    #         # pdb.set_trace()
    #
    #         # with autocast(enabled=False):
    #         f_ref1 = self.bottlneck1(torch.cat((f4, f_ref1), dim=1))
    #         offset1 = self.off2d_1(f_ref1) + flow_ref1_example.repeat(1, 72, 1, 1)
    #         f_ref1 = self.deconv_1(f_ref1.float(), offset1.float())
    #         offset2 = self.off2d_2(f_ref1)
    #         f_ref1 = self.deconv_2(f_ref1.float(), offset2.float())
    #         # offset3 = self.off2d_3(f_ref1) + flow_ref1_example.repeat(1, 72, 1, 1)
    #         # f_ref1 = self.deconv_3(f_ref1.float(), offset3.float())
    #         # offset4 = self.off2d_4(f_ref1) + flow_ref1_example.repeat(1, 72, 1, 1)
    #         # f_ref1 = self.deconv_4(f_ref1.float(), offset4.float())
    #
    #         f_ref2 = self.bottlneck1(torch.cat((f4, f_ref2), dim=1))
    #         offset1 = self.off2d_1(f_ref2) + flow_ref2_example.repeat(1, 72, 1, 1)
    #         f_ref2 = self.deconv_1(f_ref2.float(), offset1.float())
    #         offset2 = self.off2d_2(f_ref2)
    #         f_ref2 = self.deconv_2(f_ref2.float(), offset2.float())
    #         # offset3 = self.off2d_3(f_ref2) + flow_ref2_example.repeat(1, 72, 1, 1)
    #         # f_ref2 = self.deconv_3(f_ref2.float(), offset3.float())
    #         # offset4 = self.off2d_4(f_ref2) + flow_ref2_example.repeat(1, 72, 1, 1)
    #         # f_ref2 = self.deconv_4(f_ref2.float(), offset4.float())
    #
    #         ## fuse
    #         f_cur = torch.cat((f4, f_ref1, f_ref2, self.skip1(f3)), dim=1)
    #         f_cur = self.relu(self.deconv1(f_cur))
    #
    #         offset3 = self.off2d_3(torch.cat((self.skip2(f2), self.skip2(f_ref1_2)), dim=1)) + flow_ref1_example2.repeat(1, 72, 1, 1)
    #         f_ref1_2 = self.deconv_3(self.skip2(f_ref1_2).float(), offset3.float())
    #         offset4 = self.off2d_4(f_ref1_2) + flow_ref1_example2.repeat(1, 72, 1, 1)
    #         f_ref1_2 = self.deconv_4(f_ref1_2.float(), offset4.float())
    #
    #         offset3 = self.off2d_3(torch.cat((self.skip2(f2), self.skip2(f_ref2_2)), dim=1)) + flow_ref2_example2.repeat(1, 72, 1, 1)
    #         f_ref2_2 = self.deconv_3(self.skip2(f_ref2_2).float(), offset3.float())
    #         offset4 = self.off2d_4(f_ref2_2) + flow_ref2_example2.repeat(1, 72, 1, 1)
    #         f_ref2_2 = self.deconv_4(f_ref2_2.float(), offset4.float())
    #
    #         f_cur = torch.cat((f_cur, f_ref1_2, f_ref2_2, self.skip2(f2)), dim=1)
    #         f_cur = self.relu(self.deconv2(f_cur))
    #         f_cur = torch.cat((f_cur, self.skip3(f1)), dim=1)
    #         f_cur = self.relu(self.deconv3(f_cur))
    #         f_cur = self.drop(self.relu(self.deconv4(f_cur)))
    #
    #         res = self.pred(f_cur) + pre_stage_res[0]
    #
    #         return res


    # def forward(self, clip):
    #     with autocast():
    #         pre_stage_res = [self.pre_stage_network(frame) for frame in clip]
    #
    #         clip = [torch.cat([pred[0], frame], dim=1) for frame, pred in zip(clip, pre_stage_res)]
    #
    #         ## feature extractor
    #         f1, f2, f3, f4 = self.feature_extract(clip[0])
    #
    #         f_ref1 = self.first_conv(clip[1])
    #         f_ref1 = self.feature_extractor.maxpool(f_ref1)
    #         f_ref1 = self.feature_extractor.layer1(f_ref1)
    #         f_ref1 = self.feature_extractor.layer2(f_ref1)
    #         f_ref1 = self.feature_extractor.layer3(f_ref1)
    #         f_ref1 = self.feature_extractor.layer4(f_ref1)
    #         f_ref1 = self.aspp(f_ref1)
    #
    #         f_ref2 = self.first_conv(clip[2])
    #         f_ref2 = self.feature_extractor.maxpool(f_ref2)
    #         f_ref2 = self.feature_extractor.layer1(f_ref2)
    #         f_ref2 = self.feature_extractor.layer2(f_ref2)
    #         f_ref2 = self.feature_extractor.layer3(f_ref2)
    #         f_ref2 = self.feature_extractor.layer4(f_ref2)
    #         f_ref2 = self.aspp(f_ref2)
    #         pdb.set_trace()
    #
    #         # with autocast(enabled=False):
    #         f_ref1 = self.bottlneck1(torch.cat((f4, f_ref1), dim=1))
    #         offset1 = self.off2d_1(f_ref1)
    #         f_ref1 = self.deconv_1(f_ref1.float(), offset1.float())
    #         offset2 = self.off2d_2(f_ref1)
    #         f_ref1 = self.deconv_2(f_ref1.float(), offset2.float())
    #         offset3 = self.off2d_3(f_ref1)
    #         f_ref1 = self.deconv_3(f_ref1.float(), offset3.float())
    #         offset4 = self.off2d_4(f_ref1)
    #         f_ref1 = self.deconv_4(f_ref1.float(), offset4.float())
    #
    #         f_ref2 = self.bottlneck1(torch.cat((f4, f_ref2), dim=1))
    #         offset1 = self.off2d_1(f_ref2)
    #         f_ref2 = self.deconv_1(f_ref2.float(), offset1.float())
    #         offset2 = self.off2d_2(f_ref2)
    #         f_ref2 = self.deconv_2(f_ref2.float(), offset2.float())
    #         offset3 = self.off2d_3(f_ref2)
    #         f_ref2 = self.deconv_3(f_ref2.float(), offset3.float())
    #         offset4 = self.off2d_4(f_ref2)
    #         f_ref2 = self.deconv_4(f_ref2.float(), offset4.float())
    #
    #         ## fuse
    #         f_cur = torch.cat((f4, f_ref1, f_ref2, self.skip1(f3)), dim=1)
    #         f_cur = self.relu(self.deconv1(f_cur))
    #         f_cur = torch.cat((f_cur, self.skip2(f2)), dim=1)
    #         f_cur = self.relu(self.deconv2(f_cur))
    #         f_cur = torch.cat((f_cur, self.skip3(f1)), dim=1)
    #         f_cur = self.relu(self.deconv3(f_cur))
    #         f_cur = self.drop(self.relu(self.deconv4(f_cur)))
    #
    #         res = self.pred(f_cur) + pre_stage_res[0][0]
    #
    #         return res


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()