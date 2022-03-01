import cv2
import numpy as np
import os
from utils import *
import matplotlib.pyplot as plt
import tifffile
from tqdm import tqdm


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


def rgb_to_value(rgb, txt='/home/ggm/WLS/semantic/PointAnno/DataProcess/potsdam/rgb_to_value.txt'):
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
    for idx, i in tqdm(enumerate(list)):
        print(i)
        img = tifffile.imread(os.path.join(img_path, i[:-9] + 'RGBIR.tif'))[:, :, :3]
        vis = tifffile.imread(os.path.join(vis_path, i))

        label = tifffile.imread(os.path.join(root_path, i))
        label = value_to_rgb(label)

        fig, axs = plt.subplots(1, 3, figsize=(14, 4))

        axs[0].imshow(img.astype(np.uint8))
        axs[0].axis("off")
        axs[1].imshow(label.astype(np.uint8))
        axs[1].axis("off")

        axs[2].imshow(vis)
        axs[2].axis("off")

        plt.suptitle(os.path.basename(i), y=0.94)
        plt.tight_layout()
        plt.show()
        plt.close()


def show_one(path='/home/ggm/WLS/semantic/dataset/potsdam/val/label/6_7_217.tif'):
        label = tifffile.imread(path)
        label = value_to_rgb(label)

        fig, axs = plt.subplots(1, 2, figsize=(14, 4))

        axs[0].imshow(label.astype(np.uint8))
        axs[0].axis("off")

        plt.tight_layout()
        plt.show()
        plt.close()


if __name__ == '__main__':
    img_path = '/home/ggm/WLS/semantic/dataset/potsdam/dataset_origin/4_Ortho_RGBIR'

    path = '/home/ggm/WLS/semantic/dataset/potsdam/dataset_origin/5_Labels_all'
    save_path = '/home/ggm/WLS/semantic/dataset/potsdam/dataset_origin/Labels'
    check_dir(save_path)

    # save_label_value(path, save_path)
    show_label(save_path, img_path, path)