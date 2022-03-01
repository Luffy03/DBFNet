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
from dataset.data_utils import label_transform, value_to_rgb


class Dataset_sec(Dataset):
    def __init__(self, opt, save_path, file_name_txt_path, flag='train', transform=None):
        # first round
        save_path = save_path + '/Point'
        # save_path = save_path + '/Second'

        # parameters from opt
        self.data_root = os.path.join(opt.data_root, opt.dataset)
        self.dataset = opt.dataset

        if flag == 'train':
            self.img_path = self.data_root + '/train/img'
            self.label_path = save_path + '/predict_train/crf/label'

        elif flag == 'val':
            self.img_path = self.data_root + '/val/img'
            self.label_path = save_path + '/predict_val/crf/label'

        elif flag == 'predict_train':
            self.img_path = self.data_root + '/train/img'
            self.label_path = self.data_root + '/train/crf/label'

        elif flag == 'test':
            self.img_path = self.data_root + '/test/img'
            self.label_path = self.data_root + '/test/label'

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
        cls_label = torch.from_numpy(cls_label).float()
        return cls_label

    def data_transform(self, img, label):
        img = img[:, :, :self.in_channels]
        img = img.astype(np.float32).transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        if len(label.shape) == 3:
            label = label[:, :, 0]
        label = label.copy()
        label = torch.from_numpy(label).long()

        return img, label

    def __getitem__(self, index):
        item = self.img_label_path_pairs[index]
        img_path, name = item

        img = util.read(img_path)[:, :, :self.in_channels]
        label = util.read(os.path.join(self.label_path, name))
        if len(label.shape) != 3:
            label = np.expand_dims(label, axis=-1)

        if self.flag == 'train':
            if self.transform is not None:
                for t in self.transform.transforms:
                    img, label = t([img, label])

        img = util.Normalize(img, flag=self.dataset)
        img, label = self.data_transform(img, label)
        return img, label, name

    def __len__(self):

        return len(self.img_label_path_pairs)


if __name__ == "__main__":

    from utils import *
    from options import *
    from torch.utils.data import DataLoader

    opt = Sec_Options().parse()
    print(opt)
    save_path = os.path.join(opt.save_path, opt.dataset)
    train_txt_path, val_txt_path, test_txt_path = create_data_path(opt)

    train_transform = transforms.Compose([
        trans.Color_Aug(),
        trans.RandomHorizontalFlip(),
        trans.RandomVerticleFlip(),
        trans.RandomRotate90(),
    ])
    dataset = Dataset_sec(opt, save_path, train_txt_path, flag='train', transform=train_transform)

    loader = DataLoader(
        dataset=dataset,
        batch_size=1, num_workers=8, pin_memory=True, drop_last=True
        )

    for i in tqdm(loader):
        img, label, name = i

        pass