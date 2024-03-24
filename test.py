# -*- coding: utf-8 -*-
import torch
from torch import nn

nn.LayerNorm()
nn.BatchNorm1d()

m = nn.BatchNorm2d(2)
print(m.bias)
print(m.weight)

input = torch.randn(3, 2, 4, 5)
print(input)

miu = torch.mean(input, dim=(0, 2, 3), keepdim=True)
sigma = torch.sqrt(torch.var(input, dim=(0, 2, 3), keepdim=True) + m.eps)
print(miu)
print(sigma)
print(m(input))
print((input - miu) / sigma)


# m = nn.BatchNorm1d(2)
# print(m.bias)
# print(m.weight)
#
# x = torch.randn(3, 2)
# print(x)
#
# print(m(x))
# miu = torch.mean(x, dim=0)
# sigma = torch.sqrt(torch.var(x, dim=0))
# print((x - miu) / sigma)

