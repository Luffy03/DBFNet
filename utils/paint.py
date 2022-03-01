import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datafiles.color_dict import color_dict


def hex_to_rgb(value):
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def create_visual_anno(anno, dict, flag=None):
    # assert np.max(anno) <= num_classes-1, "only %d classes are supported, add new color in label2color_dict" % (num_classes)

    if flag == 'hex':
        rgb_dict = {}
        for keys, hex_value in dict.items():
            rgb_value = hex_to_rgb(hex_value)
            rgb_dict[keys] = rgb_value
    else:
        rgb_dict = dict

    # visualize
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):  # i for h
        for j in range(visual_anno.shape[1]):
            color = rgb_dict[anno[i, j]]
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]

    return visual_anno

