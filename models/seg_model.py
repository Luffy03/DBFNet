import torch
import torch.nn as nn
import torch.nn.functional as F
from models.network import *
from options import *
from models.base_model import build_base_model, build_channels
from models.tools import *


class seg_decoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(seg_decoder, self).__init__()
        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.last_layer(x)
        return x


class Seg_Net(nn.Module):
    def __init__(self, opt):
        super(Seg_Net, self).__init__()
        self.base_model = build_base_model(opt)
        self.in_channels = build_channels(opt)
        self.seg_decoder = seg_decoder(self.in_channels, opt.num_classes)
        self.img_size = opt.img_size

    def resize_out(self, output):
        if output.size()[-1] != self.img_size:
            output = F.interpolate(output, size=(self.img_size, self.img_size), mode='bilinear')
        return output

    def forward(self, x):
        x = self.base_model(x)
        # feat = x

        x = self.seg_decoder(x)
        x = self.resize_out(x)

        return x


if __name__ == "__main__":
    opt = Seg_Options().parse()
    net = Seg_Net(opt)
    net.cuda()
    x = torch.rand(4, 3, 512, 512).cuda()
    label = torch.randint(0, 8, (4, 512, 512)).cuda()
    output = net(x)

    criterion = nn.CrossEntropyLoss(ignore_index=5)
    loss = criterion(output, label)

    print(output.shape)
    print(loss)