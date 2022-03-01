import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.network.LinkNet import *


class FCN(nn.Module):
    def __init__(self, opt):
        super(FCN, self).__init__()
        if 'resnet50' in opt.backbone:
            self.backbone = ResNet50(pretrained=True)
            filters = [256, 512, 1024, 2048]

        elif 'resnet101' in opt.backbone:
            self.backbone = opt.ResNet101(pretrained=True)
            filters = [256, 512, 1024, 2048]

        elif 'resnet34' in opt.backbone:
            self.backbone = ResNet34(pretrained=True)
            filters = [64, 128, 256, 512]

        elif 'resnet18' in opt.backbone:
            self.backbone = ResNet18(pretrained=True)
            filters = [64, 128, 256, 512]
        else:
            filters = None

        self.decoder4_master = DecoderBlock(filters[3], filters[2])
        self.decoder3_master = DecoderBlock(filters[2], filters[1])
        self.decoder2_master = DecoderBlock(filters[1], filters[0])
        self.decoder1_master = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1_master = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1_master = nonlinearity

    def forward(self, x):
        x = self.backbone(x)

        # center_master
        e4 = x[3]
        # decoder_master
        d4 = self.decoder4_master(e4)
        d3 = self.decoder3_master(d4)
        d2 = self.decoder2_master(d3)
        d1 = self.decoder1_master(d2)

        out = self.finaldeconv1_master(d1)
        out = self.finalrelu1_master(out)

        return out


class UNet(nn.Module):
    def __init__(self, opt):
        super(UNet, self).__init__()
        if 'resnet50' in opt.backbone:
            self.backbone = ResNet50(pretrained=True)
            filters = [256, 512, 1024, 2048]

        elif 'resnet101' in opt.backbone:
            self.backbone = opt.ResNet101(pretrained=True)
            filters = [256, 512, 1024, 2048]

        elif 'resnet34' in opt.backbone:
            self.backbone = ResNet34(pretrained=True)
            filters = [64, 128, 256, 512]

        elif 'resnet18' in opt.backbone:
            self.backbone = ResNet18(pretrained=True)
            filters = [64, 128, 256, 512]
        else:
            filters = None

        self.decoder4_master = DecoderBlock(filters[3], filters[2])
        self.decoder3_master = DecoderBlock(filters[2]*2, filters[1])
        self.decoder2_master = DecoderBlock(filters[1]*2, filters[0])
        self.decoder1_master = DecoderBlock(filters[0]*2, filters[0])

        self.finaldeconv1_master = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1_master = nonlinearity

    def forward(self, x):
        x = self.backbone(x)

        # center_master
        e4 = x[3]
        # decoder_master
        d4 = torch.cat([self.decoder4_master(e4), x[2]], dim=1)
        d3 = torch.cat([self.decoder3_master(d4), x[1]], dim=1)
        d2 = torch.cat([self.decoder2_master(d3), x[0]], dim=1)
        d1 = self.decoder1_master(d2)

        out = self.finaldeconv1_master(d1)
        out = self.finalrelu1_master(out)


        return out