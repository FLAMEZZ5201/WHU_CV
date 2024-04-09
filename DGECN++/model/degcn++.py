
import torch
import torch.nn as nn
import darknet
from corr import Pose2DLayer
from seg  import PoseSegLayer
from resnet import ResnetEncoder
from KFA import KFA
from darknet import  Darknet
from utils import *
from DRN import DRN


class dgecn(nn.Module):
    def __init__(self,theta, data_options):
        super(dgecn, self).__init__()

        pose_arch_cfg = data_options['pose_arch_cfg']
        self.width = int(data_options['width'])
        self.height = int(data_options['height'])
        self.channels = int(data_options['channels'])

        # note you need to change this after modifying the network
        self.output_h = 76
        self.output_w = 76

        self.encoder = ResnetEncoder(18, False)
        self.DRN = DRN(theta, self.width,self.height, self.output_h, self.output_w)
        self.KFA = KFA(self.output_h, self.output_w)
        self.coreModel = Darknet(pose_arch_cfg, self.width, self.height, self.channels)
        self.segLayer = PoseSegLayer(data_options)
        self.regLayer = Pose2DLayer(data_options)
        self.training = False

    def forward(self, x, y=None):
        if self.training:
            feature = self.encoder(x)
            depth = self.KFA(20,x, feature)
            outlayers = self.coreModel(x + depth[0][0])
            out1 = self.segLayer(outlayers[0])
            out2 = self.regLayer(outlayers[1])
            out_preds = [out1, out2]
            return out_preds
        else:
            feature = self.encoder(x)
            depth_pred = self.depthlayer(feature)
            disp = depth_pred[("disp", 0)]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            disp_resized = torch.nn.functional.interpolate(
                disp, (480, 640), mode="bilinear", align_corners=False)

            outlayers = self.coreModel(x + depth[0][0])
            out1 = self.segLayer(outlayers[0])
            out2 = self.regLayer(outlayers[1])
            out_preds = [out1, out2]
            return out_preds, disp_resized

