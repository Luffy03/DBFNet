import torch
import torch.nn as nn
import torch.nn.functional as F
from models.network import *
from options import *
from models.base_model import *
from models.tools import *
import os


def clean_mask(mask, cls_label):
    n, c = cls_label.size()
    """Remove any masks of labels that are not present"""
    return mask * cls_label.view(n, c, 1, 1)


def get_penalty(predict, cls_label):
    # cls_label: (n, c)
    # predict: (n, c, h, w)
    n, c, h, w = predict.size()
    predict = torch.softmax(predict, dim=1)

    # if a patch does not contain label c,
    # then none of the pixels in this patch can be assigned to label c
    loss0 = - (1 - cls_label.view(n, c, 1, 1)) * torch.log(1 - predict + 1e-6)
    loss0 = torch.mean(torch.sum(loss0, dim=1))

    # if a patch has only one type, then the whole patch should be assigned to this type
    sum = (torch.sum(cls_label, dim=-1, keepdim=True) == 1)
    loss1 = - (sum * cls_label).view(n, c, 1, 1) * torch.log(predict + 1e-6)
    loss1 = torch.mean(torch.sum(loss1, dim=1))

    # # if a patch do has label c, then at least one pixel should be assigned to this type
    # patch_max = predict.view(n, c, -1).max(-1)[0]
    # loss2 = - cls_label * torch.log(patch_max)
    # loss2 = torch.mean(torch.sum(loss2, dim=1))

    return loss0 + loss1


def get_soft_loss(coarse_mask, soft_label):
    # mask must be softmax
    mask = F.softmax(coarse_mask, dim=1)

    loss = - soft_label * torch.log(mask + 1e-6)
    # avoid over-fitting when using soft-label, encourage the logits to 0 or 1
    penal = - mask * torch.log(mask + 1e-6)
    return loss.mean() + penal.mean()


def build_channels(backbone):
    if backbone == 'resnet34' or backbone == 'resnet18':
        channels = [64, 128, 256, 512]

    else:
        channels = [256, 512, 1024, 2048]

    return channels


def get_numiter_dilations(flag):
    if flag == 'train':
        num_iter = 1
        # for potsdam
        dilations = [[1, 3, 5, 7], [1, 3, 5], [1, 3], [1]]
    else:
        num_iter = 3
        dilations = [[1], [1], [1], [1]]
    return num_iter, dilations


class DR(nn.Module):
    def __init__(self, in_c, out_c, num_iter, dilations):
        super(DR, self).__init__()
        self.num_iter = num_iter
        self.conv = nn.Sequential(
                        nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(out_c),
                        nn.ReLU(),)
        self.DLA_m = DLAModule(num_iter=num_iter, dilations=dilations)

    def forward(self, x):
        x = self.conv(x)
        x = self.DLA_m(x)

        return x


class MultiDecoder(nn.Module):
    def __init__(self, fc, channels, flag, key_channels=96):
        super(MultiDecoder, self).__init__()
        self.fc = fc
        num_iter, dilations = get_numiter_dilations(flag)
        # dilations !!!
        self.dr0 = DR(channels[0], key_channels, num_iter, dilations[0])
        self.dr1 = DR(channels[1], key_channels, num_iter, dilations[1])
        self.dr2 = DR(channels[2], key_channels, num_iter, dilations[2])
        self.dr3 = DR(channels[3], key_channels, num_iter, dilations[3])
        self.last_conv = nn.Sequential(nn.Conv2d(key_channels*4, self.fc, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(self.fc),
                                       nn.ReLU(),
                                       nn.Conv2d(self.fc, self.fc, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(self.fc),
                                       nn.ReLU(),
                                       )

        self._init_weight()
        self.DLA_m = DLAModule(num_iter=num_iter, dilations=[1])

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
        # feat = x
        x = self.DLA_m(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ELANet(nn.Module):
    def __init__(self, opt, flag='train'):
        super(ELANet, self).__init__()
        # stride as 32
        self.backbone = build_backbone(opt.backbone, output_stride=32)
        self.img_size = opt.img_size
        self.channels = build_channels(opt.backbone)
        self.MultiD = MultiDecoder(fc=128, channels=self.channels, flag=flag)

        num_classes = opt.num_classes
        in_channels = 128
        self.seg_decoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1),
        )
        # self._ref = RefineModule(num_iter=10, dilations=[1])

    def resize_out(self, output):
        if output.size()[-1] != self.img_size:
            output = F.interpolate(output, size=(self.img_size, self.img_size), mode='bilinear')
        return output

    def forward(self, x_raw):
        features = self.backbone(x_raw)
        x = self.MultiD(features)
        x = self.seg_decoder(x)

        # use low_raw to refine the output
        # ref = RefineModule(num_iter=10, dilations=[1])
        # n, c, h, w = x.size()
        # low_raw = F.interpolate(x_raw, size=(h, w), mode='bilinear')
        # x = ref(low_raw, x)

        x = self.resize_out(x)
        return x

    def forward_loss(self, x, label, cls_label):
        coarse_mask = self.forward(x)

        # get loss
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        # penalty
        penalty = get_penalty(coarse_mask, cls_label)
        # label: point annotations point-level supervision
        seg_loss = criterion(coarse_mask, label)

        return seg_loss, penalty

    def forward_iter_loss(self, x, label, cls_label, soft_label, flag=False):
        coarse_mask = self.forward(x)

        criterion = nn.CrossEntropyLoss(ignore_index=255)
        penalty = get_penalty(coarse_mask, cls_label)
        seg_loss = criterion(coarse_mask, label)

        if flag is True:
            pse_loss = get_soft_loss(coarse_mask, soft_label)
            return seg_loss, penalty, pse_loss
        else:
            return seg_loss, penalty


if __name__ == "__main__":
    import utils.util as util
    from options import *
    from torch.utils.data import DataLoader
    from dataset import *
    from tqdm import tqdm
    from torchvision import transforms

    opt = Point_Options().parse()
    save_path = os.path.join(opt.save_path, opt.dataset)
    train_txt_path, val_txt_path, test_txt_path = util.create_data_path(opt)
    log_path, checkpoint_path, predict_path, _, _ = util.create_save_path(opt)

    train_transform = transforms.Compose([
        # trans.Color_Aug(),
        trans.RandomHorizontalFlip(),
        trans.RandomVerticleFlip(),
        trans.RandomRotate90(),
    ])
    dataset = Dataset_point(opt, train_txt_path, flag='train', transform=train_transform)

    loader = DataLoader(
        dataset=dataset,
        batch_size=2, num_workers=8, pin_memory=True, drop_last=True
    )
    net = ELANet(opt)
    net.cuda()

    for i in tqdm(loader):
        # put the data from loader to cuda
        img, label, cls_label, name = i
        input, label, cls_label = img.cuda(non_blocking=True), \
                                  label.cuda(non_blocking=True), cls_label.cuda(non_blocking=True)

        # model forward
        out = net.forward(input)
        print(out.shape)
        pass