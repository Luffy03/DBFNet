import os
import cv2
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import check_dir, read, imsave


def read_cls_color(cls):
    label2color_dict = {
        0: [255, 255, 255],  # Impervious surfaces (RGB: 255, 255, 255)
        1: [0, 0, 255],  # Building (RGB: 0, 0, 255)
        2: [0, 255, 255],  # Low vegetation (RGB: 0, 255, 255)
        3: [0, 255, 0],  # Tree (RGB: 0, 255, 0)
        4: [255, 255, 0],  # Car (RGB: 255, 255, 0)
        5: [255, 0, 0],  # Clutter/background (RGB: 255, 0, 0)
    }
    return label2color_dict[cls]


def draw_point(label, kernal_size=100, point_size=3):
    h, w, c = label.shape
    label_set = np.unique(label)

    new_mask = np.ones([h, w, c], np.uint8) * 255
    new_mask_vis = np.zeros([h, w, c], np.uint8)

    for cls in label_set:
        if cls != 255:
            color = read_cls_color(cls)

            temp_mask = np.zeros([h, w])
            temp_mask[label[:, :, 0] == cls] = 255
            temp_mask = np.asarray(temp_mask, dtype=np.uint8)
            _, contours, hierarchy = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # cv2.drawContours(new_mask, contours, -1, color, 1)

            for i in range(len(contours)):
                area = cv2.contourArea(contours[i])
                if area > kernal_size:
                    # distance to contour
                    dist = np.empty([h, w], dtype=np.float32)
                    for h_ in range(h):
                        for w_ in range(w):
                            dist[h_, w_] = cv2.pointPolygonTest(contours[i], (w_, h_), True)

                    # make sure the point in the temp_mask
                    temp_dist = temp_mask * dist
                    min_, max_, _, maxdistpt = cv2.minMaxLoc(temp_dist)
                    cx, cy = maxdistpt[0], maxdistpt[1]

                    new_mask[cy:cy + point_size, cx:cx + point_size, :] = (cls, cls, cls)
                    new_mask_vis[cy:cy + point_size, cx:cx + point_size, :] = color

    return new_mask, new_mask_vis


def make(root_path):
    train_path = root_path + '/train'
    val_path = root_path + '/val'

    paths = [train_path, val_path]
    for path in paths:
        label_path = path + '/label'

        point_label_path = path + '/point_label'
        point_label_vis_path = path + '/point_label_vis'
        check_dir(point_label_path), check_dir(point_label_vis_path)

        list = os.listdir(label_path)
        for i in tqdm(list):
            label = os.path.join(label_path, i)
            label = read(label)
            new_mask, new_mask_vis = draw_point(label)
            imsave(point_label_path + '/' + i, new_mask)
            imsave(point_label_vis_path + '/' + i, new_mask_vis)


if __name__ == '__main__':
    root_path = '/media/hlf/Luffy/WLS/semantic/dataset/potsdam'
    make(root_path)
