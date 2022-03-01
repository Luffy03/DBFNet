import os
import tifffile
import numpy as np
import cv2
from utils import check_dir, read

train_name = [7, 8, 9, 10, 11, 12]


def crop(label, vis, name, label_path, vis_path, size=256):
    height, width, _ = label.shape
    h_size = height // size
    w_size = width // size

    count = 0

    for i in range(h_size):
        for j in range(w_size):
            gt = label[i * size:(i + 1) * size, j * size:(j + 1) * size, :]
            v = vis[i * size:(i + 1) * size, j * size:(j + 1) * size, :]

            tifffile.imsave(label_path + '/' + str(name) + '_' + str(count) + '.tif', gt)
            tifffile.imsave(vis_path + '/' + str(name) + '_' + str(count) + '.tif', v)

            count += 1
    print(count)


def save(label, vis, name, save_path):
    label_path = save_path + '/label_noB'
    vis_path = save_path + '/label_vis_noB'
    check_dir(label_path), check_dir(vis_path)

    crop(label, vis, name, label_path, vis_path)


def run(root_path):
    label_path = root_path + '/dataset_origin/Labels_noB'
    vis_path = root_path + '/dataset_origin/5_Labels_all_noBoundary'

    list = os.listdir(label_path)
    for i in list:
        label = read(os.path.join(label_path, i))
        vis = read(os.path.join(vis_path, i))

        name = i[12:16]

        if int(i[14:-21]) not in train_name:
            save_path = root_path + '/test'
            check_dir(save_path)
            save(label, vis, name, save_path)


if __name__ == '__main__':
    root_path = '/home/ggm/WLS/semantic/dataset/potsdam'
    run(root_path)