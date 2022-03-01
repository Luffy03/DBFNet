# -*- coding:utf-8 -*-

# @Filename: LinkNet
# @Project : Glory
# @date    : 2020-12-28 21:29
# @Author  : Linshan

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from functools import partial
from models.network.resnet import *

nonlinearity = partial(F.relu, inplace=True)


class Dblock_more_dilate(nn.Module):
    def __init__(self, channel):
        super(Dblock_more_dilate, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out


class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out  # + dilate5_out
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class DLinkNet(nn.Module):
    def __init__(self, opt):
        super(DLinkNet, self).__init__()
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

        self.dblock_master = Dblock(filters[3])

        self.decoder4_master = DecoderBlock(filters[3], filters[2])
        self.decoder3_master = DecoderBlock(filters[2], filters[1])
        self.decoder2_master = DecoderBlock(filters[1], filters[0])
        self.decoder1_master = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1_master = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1_master = nonlinearity

    def forward(self, x):
        x = self.backbone(x)

        # center_master
        e4 = self.dblock_master(x[3])
        # decoder_master
        d4 = self.decoder4_master(e4)
        d3 = self.decoder3_master(d4)
        d2 = self.decoder2_master(d3)
        d1 = self.decoder1_master(d2)

        out = self.finaldeconv1_master(d1)
        out = self.finalrelu1_master(out)

        return out


if __name__ == '__main__':

    model = DLinkNet()

    x = torch.rand(2, 4, 512, 512)
    x = model(x)

    print(x.shape)