# -*- coding: utf-8 -*-
# @Time    : 2021/11/26 20:27
# @Author  : Jinwu Hu
# @FileName: Contrastive_head.py
# @Blog    :https://blog.csdn.net/Fhujinwu

import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    """
    The projection head used by contrast learning.
    Args:
        dim_in (int): The dimensions of input features.
        proj_dim (int): The output dimensions of projection head. Default: 256.
        proj (str): The type of projection head, only support 'linear' and 'convmlp'. Default: 'convmlp'
    """
    def __init__(self, dim=32):
        super(ProjectionHead, self).__init__()

        self.pro = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dim, out_channels=dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=dim, out_channels=dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=dim, out_channels=dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=(1,1))
        )

    def forward(self, x):
        proj = self.pro(x)
        return F.normalize(proj, p=2, dim=1)