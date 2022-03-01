# -*- coding:utf-8 -*-

# @Filename: DeeplabV3
# @Project : Glory
# @date    : 2020-12-28 21:52
# @Author  : Linshan

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from models.network.resnet import *
from models.network.segformer import *
''' 
-> ResNet BackBone
'''


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


''' 
-> The Atrous Spatial Pyramid Pooling
'''


def assp_branch(in_channels, out_channles, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channles, kernel_size, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channles),
        nn.ReLU(inplace=True))


class ASSP(nn.Module):
    def __init__(self, in_channels, aspp_stride):
        super(ASSP, self).__init__()

        assert aspp_stride in [8, 16], 'Only output strides of 8 or 16 are suported'
        if aspp_stride == 16:
            dilations = [1, 6, 12, 18]
        elif aspp_stride == 8:
            dilations = [1, 12, 24, 36]

        self.aspp1 = assp_branch(in_channels, 256, 1, dilation=dilations[0])
        self.aspp2 = assp_branch(in_channels, 256, 3, dilation=dilations[1])
        self.aspp3 = assp_branch(in_channels, 256, 3, dilation=dilations[2])
        self.aspp4 = assp_branch(in_channels, 256, 3, dilation=dilations[3])

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(256 * 5, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        initialize_weights(self)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        x5 = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
        x = self.bn1(x)
        x = self.dropout(self.relu(x))

        return x


''' 
-> Decoder
'''


class Decoder(nn.Module):
    def __init__(self, low_level_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)

        # Table 2, best performance with two 3x3 convs
        self.output = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        initialize_weights(self)

    def forward(self, x, low_level_features):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.relu(self.bn1(low_level_features))
        H, W = low_level_features.size(2), low_level_features.size(3)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.output(torch.cat((low_level_features, x), dim=1))
        return x


'''
-> Deeplab V3 +
'''


class DeepLab(nn.Module):
    def __init__(self, backbone, in_channels, pretrained=True,
                 output_stride=16, aspp_stride=8, freeze_bn=False, **_):

        super(DeepLab, self).__init__()
        assert ('resnet' or 'resnest' in backbone)

        if 'resnet18' in backbone:
            self.backbone = ResNet18(output_stride=output_stride, pretrained=pretrained, in_c=in_channels)
            low_level_channels = 64
            aspp_channels = 512

        elif 'resnet34' in backbone:
            self.backbone = ResNet34(output_stride=output_stride, pretrained=pretrained, in_c=in_channels)
            low_level_channels = 64
            aspp_channels = 512

        elif 'resnet50' in backbone:
            self.backbone = ResNet50(output_stride=output_stride, pretrained=pretrained, in_c=in_channels)
            low_level_channels = 256
            aspp_channels = 2048

        elif 'resnet101' in backbone:
            self.backbone = ResNet101(output_stride=output_stride, pretrained=pretrained, in_c=in_channels)
            low_level_channels = 256
            aspp_channels = 2048

        elif 'mit_b0' in backbone:
            self.backbone = mit_b0()
            checkpoint = torch.load('/media/hlf/Luffy/WLS/ContestCD/pretrained_former/segformer_b0.pth', map_location=torch.device('cpu'))
            self.backbone.load_state_dict(checkpoint['state_dict'])
            print('load pretrained transformer')
            low_level_channels = 32
            aspp_channels = 256

        elif 'mit_b3' in backbone:
            self.backbone = mit_b3()
            checkpoint = torch.load('/media/hlf/Luffy/WLS/ContestCD/pretrained_former/segformer_b3.pth', map_location=torch.device('cpu'))
            self.backbone.load_state_dict(checkpoint['state_dict'])
            print('load pretrained transformer')
            low_level_channels = 64
            aspp_channels = 512

        else:
            low_level_channels = None
            aspp_channels = None

        self.ASSP = ASSP(in_channels=aspp_channels, aspp_stride=aspp_stride)

        self.decoder = Decoder(low_level_channels)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.backbone(x)

        feature = self.ASSP(x[3])
        x = self.decoder(feature, x[0])
        # x = F.interpolate(x, size=(h, w), mode='bilinear')

        return x

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


if __name__ == '__main__':

    model = DeepLab(num_classes=4)

    x = torch.rand(2, 4, 256, 256)
    x = model(x)

    print(x.shape)