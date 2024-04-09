import os.path
from PIL import Image
import sys
import torchvision.transforms as transforms
import torch
from torch import nn
# 图片的切块
from  LSHDNet import  LSHDNet
import torch.nn.functional as F

def cut_image(image, patch_num):
    width, height = image.size
    item_width = int(width / patch_num)
    box_list = []
    # (left, upper, right, lower)
    for i in range(0, patch_num):  # 两重循环，生成n张图片基于原图的位置
        for j in range(0, patch_num):
            # print((i*item_width,j*item_width,(i+1)*item_width,(j+1)*item_width))
            box = (j * item_width, i * item_width, (j + 1) * item_width, (i + 1) * item_width)
            box_list.append(box)
    print(box_list)
    image_list = [image.crop(box) for box in box_list]  # Image.crop(left, up, right, below)
    return image_list

class SHPS(nn.Module):
    def __init__(self, block, layers,input_size, patch):
        super(SHPS, self).__init__()
        self.low = LSHDNet(block, layers,input_size)
        self.patch = patch

    def forward(self, x):
        low = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        low = self.low(low)
        high = F.interpolate(low , scale_factor=1, mode='bilinear', align_corners=False)
        list = cut_image(high, self.patch)






