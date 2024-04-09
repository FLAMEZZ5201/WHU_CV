import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


class KFA(nn.Module):
    def __init__(self, in_hid, out_hid):
        super(KFA, self).__init__()
        self.mlp = torch.nn.Linear(in_hid, out_hid)

    def forward(self, k, depth, color):
        d = knn(depth, k)
        c = knn(color, k)
        fuse = d + c
        out = self.mlp(fuse)

        return  out