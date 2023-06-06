# -*- coding: utf-8 -*-
# @Time    : 2021/11/10 19:33
# @Author  : Jinwu Hu
# @FileName: BasicConv.py
# @Blog    :https://blog.csdn.net/Fhujinwu
import torch.nn as nn

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x