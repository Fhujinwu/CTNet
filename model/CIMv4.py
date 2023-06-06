# -*- coding: utf-8 -*-
# @Time    : 2021/11/18 20:23
# @Author  : Jinwu Hu
# @FileName: CIMv4.py
# @Blog    :https://blog.csdn.net/Fhujinwu
import torch.nn as nn
from model.BasicConv import BasicConv2d

class CIModule(nn.Module):
    def __init__(self,out_channel=32):
        super(CIModule, self).__init__()

        self.uper =  nn.Sequential(nn.ConvTranspose2d(in_channels=out_channel, out_channels=out_channel, kernel_size=4, stride=2, padding=1),
                                   BasicConv2d(in_planes=out_channel, out_planes=out_channel,kernel_size=3,stride=1,padding=1))

    def forward(self, h, m, l):

        m_0 = self.uper(h) * m + m
        l_0 = self.uper(m) * l + l

        result = self.uper(m_0) * l_0 + l_0

        return result