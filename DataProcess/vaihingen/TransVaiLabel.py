import cv2
import numpy as np
import os
from utils import *
import matplotlib.pyplot as plt
import tifffile


def value_to_rgb(anno):
    label2color_dict = {
        0: [255, 255, 255], # Impervious surfaces (RGB: 255, 255, 255)
        1: [0, 0, 255], # Building (RGB: 0, 0, 255)
        2: [0, 255, 255], # Low vegetation (RGB: 0, 255, 255)
        3: [0, 255, 0], # Tree (RGB: 0, 255, 0)
        4: [255, 255, 0], # Car (RGB: 255, 255, 0)
        5: [255, 0, 0], # Clutter/background (RGB: 255, 0, 0)
        255: [0, 0, 0], # boundary (RGB: 0, 0, 0)
    }
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


def rgb_to_value(rgb, txt='/media/hlf/Luffy/WLS/PointAnno/DataProcess/vaihingen/rgb_to_value.txt'):
    key_arr = np.loadtxt(txt)
    array = np.zeros((rgb.shape[0], rgb.shape[1], 3), dtype=np.uint8)

    for translation in key_arr:
        r, g, b, value = translation
        tmp = [r, g, b]
        array[(rgb == tmp).all(axis=2)] = value

    return array


def save_label_value(label_path, save_path):
    check_dir(save_path)
    label_list = os.listdir(label_path)
    for idx, i in enumerate(label_list):
        print(idx)
        path = os.path.join(label_path, i)
        label = tifffile.imread(path)
        label = ((label > 128) * 255)

        label_value = rgb_to_value(label)
        tifffile.imsave(os.path.join(save_path, i), label_value)


def show_label(root_path, img_path, vis_path):
    list = os.listdir(root_path)
    for idx, i in enumerate(list):
        label = tifffile.imread(os.path.join(root_path, i))
        print(np.unique(label))
        label = value_to_rgb(label)

        img = tifffile.imread(os.path.join(img_path, i))[:, :, :3]
        vis = tifffile.imread(os.path.join(vis_path, i))

        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.subplot(1, 3, 2)
        plt.imshow(label)
        plt.subplot(1, 3, 3)
        plt.imshow(vis)
        plt.show()


if __name__ == '__main__':
    img_path = '/media/hlf/Luffy/WLS/semantic/dataset/vaihingen/dataset_origin/image'
    path = '/media/hlf/Luffy/WLS/semantic/dataset/vaihingen/dataset_origin/vis_noB'
    save_path = '/media/hlf/Luffy/WLS/semantic/dataset/vaihingen/dataset_origin/gts_noB'

    save_label_value(path, save_path)
    show_label(save_path, img_path, path)