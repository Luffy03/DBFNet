import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
import json
import cv2
import os
from utils.paint import create_visual_anno
from pathlib import Path
from models import *
from PIL import Image
import tifffile
import numpy as np
import ttach as tta
from math import *


def read(path):
    if path.endswith('.tif'):
        return tifffile.imread(path)
    else:
        img = Image.open(path)
        return np.asarray(img)


def imsave(path, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path[:-4] + '.png', img)


def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(learning_rate, optimizer, step, length, num_epochs=20):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    stride = num_epochs * length
    lr = learning_rate * (0.1 ** (step // stride))
    if step % stride == 0:
        print("learning_rate change to:%.8f" % (lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_pred_anno(out, save_pred_dir, filename, dict, flag=False):
    out = out.squeeze(0)
    check_dir(save_pred_dir)

    # save predict dir
    save_value_dir = save_pred_dir + '/label'
    check_dir(save_value_dir)
    save_value = os.path.join(save_value_dir, filename[0].split('/')[-1])
    label = out.data.cpu().numpy()
    cv2.imwrite(save_value, label)

    if flag is True:
        # save predict_visual dir
        save_anno_dir = save_pred_dir + '/label_vis'
        check_dir(save_anno_dir)
        save_anno = os.path.join(save_anno_dir, filename[0].split('/')[-1])

        img_visual = create_visual_anno(out.data.cpu().numpy(), dict=dict, flag=flag)
        b, g, r = cv2.split(img_visual)
        img_visual_rgb = cv2.merge([r, g, b])
        cv2.imwrite(save_anno, img_visual_rgb)


def save_pred_anno_numpy(out, save_pred_dir, filename, dict, flag=False):
    check_dir(save_pred_dir)

    # save predict dir
    save_value_dir = save_pred_dir + '/label'
    check_dir(save_value_dir)
    save_value = os.path.join(save_value_dir, filename[0].split('/')[-1])

    cv2.imwrite(save_value, out)

    if flag is True:
        # save predict_visual dir
        save_anno_dir = save_pred_dir + '/label_vis'
        check_dir(save_anno_dir)
        save_anno = os.path.join(save_anno_dir, filename[0].split('/')[-1])

        img_visual = create_visual_anno(out, dict=dict, flag=flag)
        b, g, r = cv2.split(img_visual)
        img_visual_rgb = cv2.merge([r, g, b])
        cv2.imwrite(save_anno, img_visual_rgb)


def save2json(metric_dict, save_path):
    file_ = open(save_path, 'w')
    file_.write(json.dumps(metric_dict, ensure_ascii=False,indent=2))
    file_.close()


def create_save_path(opt):
    save_path = os.path.join(opt.save_path, opt.dataset)
    exp_path = os.path.join(save_path, opt.experiment_name)

    log_path = os.path.join(exp_path, 'log')
    checkpoint_path = os.path.join(exp_path, 'checkpoint')
    predict_test_path = os.path.join(exp_path, 'predict_test')
    predict_train_path = os.path.join(exp_path, 'predict_train')
    predict_val_path = os.path.join(exp_path, 'predict_val')

    check_dir(save_path), check_dir(exp_path), check_dir(log_path), check_dir(checkpoint_path), \
    check_dir(predict_test_path), check_dir(predict_train_path), check_dir(predict_val_path)

    return log_path, checkpoint_path, predict_test_path, predict_train_path, predict_val_path


def create_data_path(opt):
    data_inform_path = os.path.join(opt.data_inform_path, opt.dataset)

    # for vai second
    # train_txt_path = os.path.join(data_inform_path, 'sec_train.txt')
    # val_txt_path = os.path.join(data_inform_path, 'sec_val.txt')

    train_txt_path = os.path.join(data_inform_path, 'seg_train.txt')
    val_txt_path = os.path.join(data_inform_path, 'seg_val.txt')
    test_txt_path = os.path.join(data_inform_path, 'seg_test.txt')

    return train_txt_path, val_txt_path, test_txt_path


def create_logger(log_path):
    time_str = time.strftime('%Y-%m-%d-%H-%M')

    log_file = '{}.log'.format(time_str)

    final_log_file = os.path.join(log_path , log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(log_path)/'scalar'/time_str
    print('=>creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(tensorboard_log_dir)


def resize_label(label, size):
    if len(label.size()) == 3:
        label = label.unsqueeze(1)
    label = F.interpolate(label, size=(size, size), mode='bilinear', align_corners=True)

    return label


def get_mean_std(flag):
    if flag == 'potsdam':
        means = [86.42521457, 92.37607528, 85.74658389]
        std = [35.58409409, 35.45218542, 36.91464009]
    elif flag == 'vaihingen':
        means = [119.14901543,  83.04203606,  81.79810095]
        std = [55.63038161, 40.67145608, 38.61447761]

    else:
        means = 0
        std = 0
        print('error')
    return means, std


def Normalize(img, flag='potsdam'):
    means, std = get_mean_std(flag)
    img = (img - means) / std

    return img


def Normalize_back(img, flag='potsdam'):
    means, std = get_mean_std(flag)

    means = means[:3]
    std = std[:3]

    img = img * std + means

    return img


def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = F.pad(img, (0, 0, rows_missing, cols_missing), 'constant', 0)
    return padded_img


def pre_slide(model, image, num_classes=7, tile_size=(512, 512), tta=False):
    image_size = image.shape  # bigger than (1, 3, 512, 512), i.e. (1,3,1024,1024)
    overlap = 1 / 2  # 每次滑动的重合率为1/2

    stride = ceil(tile_size[0] * (1 - overlap))  # 滑动步长:769*(1-1/3) = 513
    tile_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)  # 行滑动步数:(1024-769)/513 + 1 = 2
    tile_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)  # 列滑动步数:(2048-769)/513 + 1 = 4

    full_probs = torch.zeros((1, num_classes, image_size[2], image_size[3])).cuda()  # 初始化全概率矩阵 shape(1024,2048,19)

    count_predictions = torch.zeros((1, 1, image_size[2], image_size[3])).cuda()  # 初始化计数矩阵 shape(1024,2048,19)
    tile_counter = 0  # 滑动计数0

    for row in range(tile_rows):  # row = 0,1
        for col in range(tile_cols):  # col = 0,1,2,3
            x1 = int(col * stride)  # 起始位置x1 = 0 * 513 = 0
            y1 = int(row * stride)  # y1 = 0 * 513 = 0
            x2 = min(x1 + tile_size[1], image_size[3])  # 末位置x2 = min(0+769, 2048)
            y2 = min(y1 + tile_size[0], image_size[2])  # y2 = min(0+769, 1024)
            x1 = max(int(x2 - tile_size[1]), 0)  # 重新校准起始位置x1 = max(769-769, 0)
            y1 = max(int(y2 - tile_size[0]), 0)  # y1 = max(769-769, 0)

            img = image[:, :, y1:y2, x1:x2]  # 滑动窗口对应的图像 imge[:, :, 0:769, 0:769]
            padded_img = pad_image(img, tile_size)  # padding 确保扣下来的图像为769*769

            tile_counter += 1  # 计数加1
            # print("Predicting tile %i" % tile_counter)

            # 将扣下来的部分传入网络，网络输出概率图。
            # use softmax
            if tta is True:
                padded = tta_predict(model, padded_img)
            else:
                padded = model(padded_img)
                padded = F.softmax(padded, dim=1)

            pre = padded[:, :, 0:img.shape[2], 0:img.shape[3]]  # 扣下相应面积 shape(769,769,19)

            count_predictions[:, :, y1:y2, x1:x2] += 1  # 窗口区域内的计数矩阵加1
            full_probs[:, :, y1:y2, x1:x2] += pre  # 窗口区域内的全概率矩阵叠加预测结果

    # average the predictions in the overlapping regions
    full_probs /= count_predictions  # 全概率矩阵 除以 计数矩阵 即得 平均概率

    return full_probs   # 返回整张图的平均概率 shape(1, 1, 1024,2048)


def tta_predict(model, img):
    tta_transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270]),
        ])

    xs = []

    for t in tta_transforms:
        aug_img = t.augment_image(img)
        aug_x = model(aug_img)
        aug_x = F.softmax(aug_x, dim=1)

        x = t.deaugment_mask(aug_x)
        xs.append(x)

    xs = torch.cat(xs, 0)
    x = torch.mean(xs, dim=0, keepdim=True)

    return x