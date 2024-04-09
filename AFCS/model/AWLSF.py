import numpy as np
import cv2
# import math
from scipy.sparse import spdiags


def wlsFilter(img, Lambda, alpha=1.2, eps=0.0001):
    L = np.log(img + 0.0001)
    row, cols = img.shape[:2]
    k = row * cols

    # 对L矩阵的第一维度上做差分，也就是下面的行减去上面的行，得到(N-1)xM维的矩阵
    dy = np.diff(L, 1, 0)
    dy = -Lambda / (np.power(np.abs(dy), alpha) + eps)
    # 在最后一行的后面补上一行0
    dy = np.pad(dy, ((0, 1), (0, 0)), 'constant')
    # 按列生成向量，对应上面Ay的对角线元素
    dy = dy.T
    dy = dy.reshape(-1, 1)

    # 对L矩阵的第二维度上做差分，也就是右边的列减去左边的列，得到Nx(M-1)的矩阵
    dx = np.diff(L, 1, 1)
    dx = -Lambda / (np.power(np.abs(dx), alpha) + eps)
    # 在最后一列的后面补上一行0
    dx = np.pad(dx, ((0, 0), (0, 1)), 'constant')
    # 按列生成向量，对应上面Ay的对角线元素
    dx = dx.T
    dx = dy.reshape(-1, 1)

    # 构造五点空间非齐次拉普拉斯矩阵
    B = np.hstack((dx, dy))
    B = B.T
    diags = np.array([-row, -1])
    # 把dx放在-row对应的对角线上，把dy放在-1对应的对角线上
    A = spdiags(B, diags, k, k).toarray()

    e = dx
    w = np.pad(dx, ((row, 0), (0, 0)), 'constant')
    w = w[0:-row]

    s = dy
    n = np.pad(dy, ((1, 0), (0, 0)), 'constant')
    n = n[0:-1]

    D = 1 - (e + w + s + n)
    D = D.T
    # A只有五个对角线上有非0元素
    diags1 = np.array([0])
    A = A + np.array(A).T + spdiags(D, diags1, k, k).toarray()

    im = np.array(img)
    p, q = im.shape[:2]
    g = p * q
    im = np.reshape(im, (g, 1))

    a = np.linalg.inv(A)

    out = np.dot(a, im)

    out = np.reshape(out, (row, cols))

    return out

