import torch.nn as nn
from  SHPS import SHPS
from RM import RM
from AFM import AFM
class PRBNet(nn.Module):
    def __init__(self, block, layers,input_size, patch,  in_channles, num_channles):
         super(PRBNet, self).__init__()
         self.sphs1 = SHPS(block, layers,input_size, patch[0]) #256
         self.sphs2 = SHPS(block, layers, input_size, patch[1]) #512
         self.sphs3 = SHPS(block, layers, input_size, patch[2]) #1024
         self.sphs3 = SHPS(block, layers, input_size, patch[3]) #2048
         self.RM = RM(in_channles, num_channles)
         self.AFM = AFM(in_channles, num_channles)

    def forward(self, x):
         s1 = self.sphs1(x)
         s2 = self.sphs2(x)
         s3 = self.sphs3(x)
         s4 = self.sphs4(x)


         # large to small
         y1 = self.RM(s4, s3)
         y2 = self.RM(y1, s2)
         y3 = self.RM(y2, s1)

         # small to large
         z1 = self.RM(s1, s2)
         z2 = self.RM(z1, s3)
         z3 = self.RM(z2, s4)


         out = self.AFM(y3, z3)
         return  out






