import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from models import ELANet, Seg_Net
from matplotlib import pyplot as plt
from utils import *
from dataset.datapoint import value_to_rgb
import cv2


def compute_saliency_maps(img, label, model, flag='seg'):
    model.eval()
    n, h, w = label.shape
    # mask
    mask = torch.zeros_like(label)
    mask[label != 255] = 1
    mask = mask.float().cuda()

    label_new = label.clone()
    label_new[label == 255] = 0

    if flag == 'seg':
        feature_map = model.base_model(img.detach())
        feature_map.retain_grad()
        logits = model.resize_out(model.seg_decoder(feature_map))
    else:
        feature_map = model.MultiD(model.backbone(img.detach()))[1]
        # retain_grad make the non-leaf to have grad
        feature_map.retain_grad()
        logits = model.resize_out(model.seg_decoder
                                  (model.MultiD.DLA_m(feature_map)))

    # img.requires_grad_()
    # logits = model.forward(img)

    logits = torch.gather(logits, dim=1, index=label_new.view(n, 1, h, w)).squeeze(0)
    logits.backward(mask)
    # saliency = abs(img.grad.data)
    saliency = abs(feature_map.grad.data)  # 返回X的梯度绝对值大小
    sal_map = saliency.sum(1).squeeze().data.cpu().numpy()
    sal_map = sal_map/sal_map.max()

    sal_map *= 10
    sal_map = (255*sal_map).astype(np.uint8)
    sal_map = np.clip(sal_map, 0, 255)
    return sal_map

# def save_saliency_maps(sal_maps):


def show_saliency_maps(img, label, true_label, sal_maps, name):
    length = len(sal_maps)
    fig, axs = plt.subplots(1, 3 + length, figsize=(14, 4))

    img = img.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = Normalize_back(img)
    axs[0].imshow(img[:, :, :3].astype(np.uint8))
    axs[0].axis("off")

    true_label = true_label.permute(1, 2, 0).cpu().numpy()
    true_vis = value_to_rgb(true_label)
    axs[1].imshow(true_vis.astype(np.uint8))
    axs[1].axis("off")

    label = label.permute(1, 2, 0).cpu().numpy()
    vis = value_to_rgb(label)
    axs[2].imshow(vis.astype(np.uint8))
    axs[2].axis("off")

    for i, sal_map in enumerate(sal_maps):
        # sal_map = cv2.applyColorMap(sal_map, cv2.COLORMAP_JET)
        axs[3+i].imshow(sal_map, cmap=plt.cm.hot)
        axs[3+i].axis("off")

    plt.suptitle(os.path.basename(name[0]), y=0.94)
    plt.tight_layout()
    plt.show()
    plt.close()


def read_full_label(filename):
    path = '/home/ggm/WLS/semantic/dataset/potsdam/train/label/'+filename[0]
    label = read(path)
    from dataset.data_utils import label_transform
    label = label_transform('potsdam', 'train', label)
    if len(label.shape) == 3:
        label = label[:, :, 0]
    label = torch.from_numpy(label).long()
    label = label.unsqueeze(0)
    return label


if __name__ == "__main__":
    import utils.util as util
    from options import *
    from torch.utils.data import DataLoader
    from dataset import *
    from tqdm import tqdm

    opt = Point_Options().parse()
    save_path = os.path.join(opt.save_path, opt.dataset)
    train_txt_path, val_txt_path, test_txt_path = util.create_data_path(opt)
    log_path, checkpoint_path, predict_path, _, _ = util.create_save_path(opt)

    dataset = Dataset_point(opt, train_txt_path, flag='train', transform=None)
    loader = DataLoader(
        dataset=dataset,
        batch_size=1, num_workers=8, pin_memory=True, drop_last=True, shuffle=True
    )

    net0 = Seg_Net(opt)
    checkpoint = torch.load('/home/ggm/WLS/semantic/PointAnno/save/potsdam/baseline/checkpoint/model_best.pth', map_location=torch.device('cpu'))
    net0.load_state_dict(checkpoint['state_dict'])
    print('resume success')
    net0.cuda()

    net1 = ELANet(opt)
    checkpoint = torch.load(checkpoint_path + '/model_best_0.8423.pth', map_location=torch.device('cpu'))
    net1.load_state_dict(checkpoint['state_dict'])
    print('resume success')
    net1.cuda()

    net2 = Seg_Net(opt)
    checkpoint = torch.load('/home/ggm/WLS/semantic/PointAnno/save/potsdam/Seg/checkpoint/model_best_0.8961.pth',
                            map_location=torch.device('cpu'))
    net2.load_state_dict(checkpoint['state_dict'])
    print('resume success')
    net2.cuda()

    for i in tqdm(loader):
        # put the data from loader to cuda
        img, label, cls_label, name = i
        label, cls_label = label.cuda(non_blocking=True), cls_label.cuda(non_blocking=True)
        input = img.cuda(non_blocking=True)

        full_label = read_full_label(name)
        full_label = full_label.cuda(non_blocking=True)

        sal_map0 = compute_saliency_maps(input, label, net0)
        sal_map1 = compute_saliency_maps(input, label, net1, flag='ELA')
        sal_map2 = compute_saliency_maps(input, full_label, net2)
        sal_maps = [sal_map0, sal_map1, sal_map2]

        show_saliency_maps(img, label, full_label, sal_maps, name)