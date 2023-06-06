# -*- coding: utf-8 -*-
# @Time    : 2021/11/22 17:00
# @Author  : Jinwu Hu
# @FileName: Mymodel13.py
# @Blog    :https://blog.csdn.net/Fhujinwu


import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MIT import mit_b3
from model.CIMv4 import CIModule
from model.BasicConv import BasicConv2d
from model.SMIMv2 import SMIModule
from model.Contrastive_head import ProjectionHead

class PolypModule(nn.Module):
    def __init__(self, channel=32):
        super(PolypModule, self).__init__()
        print("Modelv13")
        self.backbone = mit_b3()  # [64, 128, 320, 512]
        path = 'xxx/pretrained_pth/mit_b3.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        print(state_dict)
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)


        self.Translayer1 = BasicConv2d(64, channel, 1)
        self.Translayer2 = BasicConv2d(128, channel, 1)
        self.Translayer3 = BasicConv2d(320, channel, 1)
        self.Translayer4 = BasicConv2d(512, channel, 1)
        self.contrasthead = ProjectionHead(dim=32)

        self.Smim = SMIModule(out_channel=channel)
        self.cim = CIModule(out_channel=channel)

        self.uper =  nn.Sequential(nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=4, stride=2, padding=1),
                                   BasicConv2d(in_planes=channel, out_planes=channel,kernel_size=3,stride=1,padding=1))

        self.prediction = nn.Sequential(BasicConv2d(in_planes=channel,out_planes=channel,kernel_size=3, stride=1,padding=1),
                                        nn.Conv2d(channel, 1, 1))


    def forward(self, x):

        #feature extraction
        cswin = self.backbone(x)

        x1 = self.Translayer1(cswin[0])
        x2 = self.Translayer2(cswin[1])
        x3 = self.Translayer3(cswin[2])
        x4 = self.Translayer4(cswin[3])

        emb = self.contrasthead(x4)

        x4 = self.Smim(x4)
        x3 = self.Smim(x3)
        x2 = self.Smim(x2)

        x_4_3_2 = self.cim(x4,x3,x2)
        x1 = x1 + self.uper(x_4_3_2)
        prediction1 = self.prediction(x1)
        prediction1_4 = F.interpolate(prediction1, scale_factor=4, mode='bilinear', align_corners=False)
        # return prediction1_4
        return prediction1_4, emb


if __name__ == '__main__':
    model = PolypModule().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction = model(input_tensor)

    print(prediction.size())
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params)