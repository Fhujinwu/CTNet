# -*- coding: utf-8 -*-
# @Time    : 2021/11/4 15:13
# @Author  : Jinwu Hu
import torch
from apex.amp import opt
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from model.Mymodel13 import PolypModule
from utils.Mydataloader import get_loader, test_dataset
from utils.utils import clip_gradient,AvgMeter,adjust_lr
import torch.nn.functional as F
import numpy as np
import logging
from utils.loss import structure_loss
from utils.pixel_contrast_cross_entropy_loss import ContrastCELoss

def train(train_loader, model, optimizer, epoch):
    model.train()
    size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            P, emb = model(images)
            loss_contrast = ContrastCELoss()(P,emb,gts)
            loss_P = structure_loss(P, gts)
            loss = loss_P + 0.1*loss_contrast
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)

        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
            format(datetime.now(), epoch, opt.epoch, i, total_step,
                   loss_record.show()))
    # save model
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypModel.pth')

if __name__ == '__main__':
    model_name = 'CTNet'
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=32, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='xxx/TrainDataset/',
                        help='path to train dataset')

    parser.add_argument('--train_save', type=str,
                        default='./model_pth/'+model_name+'/')

    opt = parser.parse_args()
    logging.basicConfig(filename='train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    model = PolypModule().cuda()

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        print(opt.lr)
        train(train_loader, model, optimizer, epoch)
