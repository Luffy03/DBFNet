import torch
import torch.nn as nn
import torch.nn.functional as F
from models.network import *
from options import *


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


def build_base_model(opt):
    # get in_channels
    if opt.base_model == 'Deeplab':
        model = DeepLab(backbone=opt.backbone, in_channels=opt.in_channels)

    elif opt.base_model == 'LinkNet':
        model = DLinkNet(opt)

    elif opt.base_model == 'FCN':
        model = FCN(opt)

    elif opt.base_model == 'UNet':
        model = UNet(opt)

    else:
        model = None
        print('model none')
    return model


def build_channels(opt):
    if opt.base_model == 'Deeplab':
        channels = 256
    elif opt.base_model == 'STANet':
        channels = 128
    elif opt.base_model == 'LinkNet':
        channels = 32
    elif opt.base_model == 'FCN':
        channels = 32
    elif opt.base_model == 'UNet':
        channels = 32
    else:
        channels = None
        print('model none')
    return channels


