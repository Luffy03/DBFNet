import torch
from torch.utils.data import Dataset
import numpy as np
import os
from utils import util
from tqdm import tqdm
import multiprocessing
from torchvision import transforms
import dataset.transform as trans
import tifffile
import matplotlib.pyplot as plt
from dataset.data_utils import label_transform


class Dataset_point(Dataset):
    def __init__(self, opt, file_name_txt_path, flag='train', transform=None):
        # parameters from opt
        self.data_root = os.path.join(opt.data_root, opt.dataset)
        self.dataset = opt.dataset

        if flag == 'test':
            self.img_path = self.data_root + '/test/img'
            if self.dataset == 'vaihingen':
                self.label_path = self.data_root + '/test/label_noB'
            else:
                self.label_path = self.data_root + '/test/label'
        elif flag == 'train':
            self.img_path = self.data_root + '/train/img'
            self.label_path = self.data_root + '/train/point_label'
        elif flag == 'val':
            self.img_path = self.data_root + '/val/img'
            self.label_path = self.data_root + '/val/point_label'

        elif flag == 'predict_train':
            self.img_path = self.data_root + '/train/img'
            self.label_path = self.data_root + '/train/label'

        self.img_size = opt.img_size
        self.num_classes = opt.num_classes
        self.in_channels = opt.in_channels

        self.img_txt_path = file_name_txt_path
        self.flag = flag
        self.transform = transform
        self.img_label_path_pairs = self.get_img_label_path_pairs()

    def get_img_label_path_pairs(self):
        img_label_pair_list = {}

        with open(self.img_txt_path, 'r') as lines:
            for idx, line in enumerate(tqdm(lines)):
                name = line.strip("\n").split(' ')[0]
                path = os.path.join(self.img_path, name)
                img_label_pair_list.setdefault(idx, [path, name])

        return img_label_pair_list

    def make_clslabel(self, label):
        label_set = np.unique(label)
        cls_label = np.zeros(self.num_classes)
        for i in label_set:
            if i != 255:
                cls_label[i] += 1
        return cls_label

    def data_transform(self, img, label, cls_label):
        img = img[:, :, :self.in_channels]
        img = img.astype(np.float32).transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        if len(label.shape) == 3:
            label = label[:, :, 0]

        label, cls_label = label.copy(), cls_label.copy()

        label = torch.from_numpy(label).long()
        cls_label = torch.from_numpy(cls_label).float()

        return img, label, cls_label

    def __getitem__(self, index):
        item = self.img_label_path_pairs[index]
        img_path, name = item

        img = util.read(img_path)[:, :, :3]
        img = util.Normalize(img, flag=self.dataset)
        label = util.read(os.path.join(self.label_path, name))

        label = label_transform(self.dataset, label)
        cls_label = self.make_clslabel(label)

        # data transform
        if self.transform is not None:
            for t in self.transform.transforms:
                img, label = t([img, label])

        img, label, cls_label = self.data_transform(img, label, cls_label)
        return img, label, cls_label, name

    def __len__(self):

        return len(self.img_label_path_pairs)


def value_to_rgb(anno, flag='potsdam'):
    if flag == 'potsdam' or flag == 'vaihingen':
        label2color_dict = {
            0: [255, 255, 255], # Impervious surfaces (RGB: 255, 255, 255)
            1: [0, 0, 255], # Building (RGB: 0, 0, 255)
            2: [0, 255, 255], # Low vegetation (RGB: 0, 255, 255)
            3: [0, 255, 0], # Tree (RGB: 0, 255, 0)
            4: [255, 255, 0], # Car (RGB: 255, 255, 0)
            5: [255, 0, 0], # Clutter/background (RGB: 255, 0, 0)
            255: [0, 0, 0]
        }
    else:
        label2color_dict = {}

    # visualize
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):  # i for h
        for j in range(visual_anno.shape[1]):
            # cv2: bgr
            color = label2color_dict[anno[i, j, 0]]

            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]

    return visual_anno


def show_pair(opt, img, label, name):
    true_label = read('/media/hlf/Luffy/WLS/semantic/dataset/'+opt.dataset+'/train/label_vis/' + name[0])
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = Normalize_back(img, flag=opt.dataset)
    axs[0].imshow(img[:, :, :3].astype(np.uint8))
    axs[0].axis("off")

    label = label.permute(1, 2, 0).cpu().numpy()
    print(np.unique(label))
    vis = value_to_rgb(label, flag=opt.dataset)
    axs[1].imshow(vis.astype(np.uint8))
    axs[1].axis("off")

    axs[2].imshow(true_label.astype(np.uint8))
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":

    from utils import *
    from options import *
    from torch.utils.data import DataLoader

    opt = Point_Options().parse()
    print(opt)
    train_txt_path, val_txt_path, test_txt_path = create_data_path(opt)

    train_transform = transforms.Compose([
        trans.Scale(opt.img_size),
        trans.RandomHorizontalFlip(),
        trans.RandomVerticleFlip(),
        trans.RandomRotate90(),
    ])
    dataset = Dataset_point(opt, train_txt_path, flag='train', transform=None)

    loader = DataLoader(
        dataset=dataset, shuffle=True,
        batch_size=1, num_workers=8, pin_memory=True, drop_last=True
        )

    for i in tqdm(loader):
        img, label, cls_label, name = i
        show_pair(opt, img, label, name)
        pass