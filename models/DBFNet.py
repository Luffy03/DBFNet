from models.base_model import *
from models.tools import *
import os
from models.fbf import FBFModule


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


def build_channels(backbone):
    if backbone == 'resnet34' or backbone == 'resnet18':
        channels = [64, 128, 256, 512]

    else:
        channels = [256, 512, 1024, 2048]

    return channels


def get_numiter_dilations(flag):
    if flag == 'train':

        # for potsdam
        num_iter = 1
        dilations = [[1, 3, 5, 7], [1, 3, 5], [1, 3], [1]]

    else:
        # for potsdam
        num_iter = 3
        dilations = [[1], [1], [1], [1]]

        # for vaihingen
        # num_iter = 5
        # dilations = [[1], [1], [1], [1]]
    return num_iter, dilations


class FBF_Layer(nn.Module):
    def __init__(self, in_c, out_c, num_iter, dilations):
        super(FBF_Layer, self).__init__()
        self.num_iter = num_iter
        self.fbf_m = FBFModule(num_iter=num_iter, dilations=dilations)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_c),
            nn.ReLU(), )

    def forward(self, x):
        # x = self.conv0(x)
        x = self.fbf_m(x)
        x = self.conv(x)

        return x


class Net(nn.Module):
    def __init__(self, opt, flag='train'):
        super(Net, self).__init__()
        # stride as 32
        self.backbone = build_backbone(opt.backbone, output_stride=32)
        self.img_size = opt.img_size

        # build DBF layer
        num_iter, dilations = get_numiter_dilations(flag)
        channels = build_channels(opt.backbone)
        key_channels = 128
        self.FBF_layer1 = FBF_Layer(channels[0], key_channels, num_iter, dilations[0])
        self.FBF_layer2 = FBF_Layer(channels[1], key_channels, num_iter, dilations[1])
        self.FBF_layer3 = FBF_Layer(channels[2], key_channels, num_iter, dilations[2])
        self.FBF_layer4 = FBF_Layer(channels[3], key_channels, num_iter, dilations[3])

        self.last_conv = nn.Sequential(
            nn.Conv2d(4*128, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            )
        self.FBF_layer = FBF_Layer(key_channels, key_channels, 1, dilations=[1])
        self.seg_decoder = nn.Sequential(
            nn.Conv2d(key_channels, key_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(),
            nn.Conv2d(key_channels, opt.num_classes, kernel_size=3, stride=1, padding=1),
        )

    def resize_out(self, output):
        if output.size()[-1] != self.img_size:
            output = F.interpolate(output, size=(self.img_size, self.img_size), mode='bilinear')
        return output

    def upsample_cat(self, p1, p2, p3, p4):
        p2 = nn.functional.interpolate(p2, size=p1.size()[2:], mode='bilinear', align_corners=True)
        p3 = nn.functional.interpolate(p3, size=p1.size()[2:], mode='bilinear', align_corners=True)
        p4 = nn.functional.interpolate(p4, size=p1.size()[2:], mode='bilinear', align_corners=True)
        return torch.cat([p1, p2, p3, p4], dim=1)

    def forward(self, x):
        # resnet 4 layers
        l1, l2, l3, l4 = self.backbone(x)

        # after feature bilateral filtering
        f1 = self.FBF_layer1(l1)
        f2 = self.FBF_layer2(l2)
        f3 = self.FBF_layer3(l3)
        f4 = self.FBF_layer4(l4)

        # fpn-like Top-down
        p4 = f4
        p3 = F.upsample(p4, size=f3.size()[2:], mode='bilinear') + f3
        p2 = F.upsample(p3, size=f2.size()[2:], mode='bilinear') + f2
        p1 = F.upsample(p2, size=f1.size()[2:], mode='bilinear') + f1

        cat = self.upsample_cat(p1, p2, p3, p4)
        feat = self.last_conv(cat)
        feat = self.FBF_layer(feat)

        out = self.seg_decoder(feat)
        out = self.resize_out(out)

        return out

    def forward_loss(self, x, label, cls_label):
        coarse_mask = self.forward(x)

        # get loss
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        # penalty
        penalty = get_penalty(coarse_mask, cls_label)
        # label: point annotations point-level supervision
        seg_loss = criterion(coarse_mask, label)

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
        trans.RandomHorizontalFlip(),
        trans.RandomVerticleFlip(),
        trans.RandomRotate90(),
    ])
    dataset = Dataset_point(opt, train_txt_path, flag='train', transform=train_transform)

    loader = DataLoader(
        dataset=dataset,
        batch_size=1, num_workers=8, pin_memory=True, drop_last=True
    )
    net = Net(opt, flag='train')
    net.cuda()
    torch.save({'state_dict': net.state_dict()},
               os.path.join(checkpoint_path, 'model.pth'))

    for i in tqdm(loader):
        # put the data from loader to cuda
        img, label, cls_label, name = i
        input, label, cls_label = img.cuda(non_blocking=True), \
                                  label.cuda(non_blocking=True), cls_label.cuda(non_blocking=True)

        # model forward
        out = net.forward(input)

        pass