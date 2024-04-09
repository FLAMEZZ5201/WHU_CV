import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, in_dim, out_dim, in_q_dim, hid_q_dim):
        super(CrossAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_q_dim = in_q_dim
        self.hid_q_dim = hid_q_dim
        self.query = nn.Linear(in_q_dim, hid_q_dim, bias=False)
        self.key = nn.Linear(in_dim, out_dim, bias=False)
        self.value = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, y):

        batch_size = x.shape[0]  # batch size
        num_queries = x.shape[1]  # 查询矩阵中的元素个数
        num_keys = y.shape[1]  # 键值矩阵中的元素个数
        x = self.query(x)  # 查询矩阵
        y = self.key(y)  # 键值矩阵
        # 计算注意力分数
        attn_scores = torch.matmul(x, y.transpose(-2, -1)) / (
                    self.out_dim ** 0.5)  # 计算注意力分数，注意力分数矩阵的大小为 batch_size x num_queries x num_keys x num_keys
        attn_weights = F.softmax(attn_scores, dim=-1)  # 对注意力分数进行 softmax 归一化
        # 计算加权和
        V = self.value(y)  # 通过值变换得到值矩阵 V
        output = torch.bmm(attn_weights, V)  # 计算加权和，output 的大小为 batch_size x num_queries x num_keys x out_dim

        return output


class DRN(nn.Module):
    def __init__(self, theta, in_dim,  out_dim, in_q_dim, hid_q_dim):
        super(DRN, self).__init__()
        self.d_x1 = torch.laod("pre_dmodel.pth")
        self.d_x2 = torch.laod("pre_dmodel.pth")
        self.ca = CrossAttention(in_dim,  out_dim, in_q_dim, hid_q_dim)
        self.mlp = torch.nn.Linear(out_dim, out_dim)
        self.theta =theta

    def forward(self, x):
        ut = torch.bool((self.d_x1(x) -self.d_x2(x))<self.theta)
        x1 = self.ca(x)
        x1 = ut*x1
        sa = self.ca(self.d_x1(x), x1)
        out = self.mlp(sa)+self.d_x1(x)
        return out




