# -*- coding:utf-8 -*-

# @Filename: STANet
# @Project : Glory
# @date    : 2020-12-28 21:52
# @Author  : Linshan

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from models.network.resnet import *


def build_backbone(backbone, output_stride, pretrained=True, in_c=3):
    if backbone == 'resnet50':
        return ResNet50(output_stride, pretrained=pretrained, in_c=in_c)

    elif backbone == 'resnet101':
        return ResNet101(output_stride, pretrained=pretrained, in_c=in_c)

    elif backbone == 'resnet34':
        return ResNet34(output_stride, pretrained=pretrained, in_c=in_c)

    elif backbone == 'resnet18':
        return ResNet18(output_stride, pretrained=pretrained, in_c=in_c)

    else:
        raise NotImplementedError


def build_channels(backbone):
    if backbone == 'resnet34' or backbone == 'resnet18':
        channels = [64, 128, 256, 512]

    else:
        channels = [256, 512, 1024, 2048]

    return channels


class DR(nn.Module):
    def __init__(self, in_d, out_d):
        super(DR, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv1 = nn.Conv2d(self.in_d, self.out_d, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_d)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, fc, BatchNorm, channels, key_channels=96):
        super(Decoder, self).__init__()
        self.fc = fc
        self.dr0 = DR(channels[0], key_channels)
        self.dr1 = DR(channels[1], key_channels)
        self.dr2 = DR(channels[2], key_channels)
        self.dr3 = DR(channels[3], key_channels)
        self.last_conv = nn.Sequential(nn.Conv2d(key_channels*4, self.fc, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(self.fc),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(self.fc, self.fc, kernel_size=1, stride=1, padding=0, bias=False),
                                       BatchNorm(self.fc),
                                       nn.ReLU(),
                                       )

        self._init_weight()

    def forward(self, x):

        x0 = self.dr0(x[0])
        x1 = self.dr1(x[1])
        x2 = self.dr2(x[2])
        x = self.dr3(x[3])

        x1 = F.interpolate(x1, size=x0.size()[2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=x0.size()[2:], mode='bilinear', align_corners=True)
        x = F.interpolate(x, size=x0.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x, x0, x1, x2), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class STANet(nn.Module):
    def __init__(self, backbone='resnet18', output_stride=16, f_c=128, freeze_bn=False, in_channels=3):
        super(STANet, self).__init__()
        BatchNorm = nn.BatchNorm2d
        channels = build_channels(backbone)

        self.backbone = build_backbone(backbone, output_stride, in_c=in_channels)
        self.decoder = Decoder(f_c, BatchNorm, channels)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x = self.backbone(input)
        x = self.decoder(x)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


if __name__ == '__main__':
    net = STANet()
    a = torch.randn(1, 4, 256, 256)
    b = net(a)
    print(b.shape)
