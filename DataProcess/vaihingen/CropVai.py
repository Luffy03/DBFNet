import os
import tifffile
import numpy as np
import cv2
from utils import check_dir, read, imsave

test_name = [2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29, 31, 33, 35, 38]
val_name = [5]


def crop_overlap(img, label, vis, name, img_path, label_path, vis_path, size=256, stride=128):
    h, w, c = img.shape
    new_h, new_w = (h // size + 1) * size, (w // size + 1) * size
    num_h, num_w = (new_h // stride) - 1, (new_w // stride) - 1

    new_img = np.zeros([new_h, new_w, c]).astype(np.uint8)
    new_img[:h, :w, :] = img

    new_label = 255 * np.ones([new_h, new_w, 3]).astype(np.uint8)
    new_label[:h, :w, :] = label

    new_vis = np.zeros([new_h, new_w, 3]).astype(np.uint8)
    new_vis[:h, :w, :] = vis

    count = 0

    for i in range(num_h):
        for j in range(num_w):
            out = new_img[i * stride:i * stride + size, j * stride:j * stride + size, :]
            gt = new_label[i * stride:i * stride + size, j * stride:j * stride + size, :]
            v = new_vis[i * stride:i * stride + size, j * stride:j * stride + size, :]
            assert v.shape == (256, 256, 3), print(v.shape)

            imsave(img_path + '/' + str(name) + '_' + str(count) + '.png', out)
            imsave(label_path + '/' + str(name) + '_' + str(count) + '.png', gt)
            imsave(vis_path + '/' + str(name) + '_' + str(count) + '.png', v)

            count += 1


def crop(img, label, vis, name, img_path, label_path, vis_path, size=256):
    height, width, _ = img.shape
    h_size = height // size
    w_size = width // size

    count = 0

    for i in range(h_size):
        for j in range(w_size):
            out = img[i * size:(i + 1) * size, j * size:(j + 1) * size, :]
            gt = label[i * size:(i + 1) * size, j * size:(j + 1) * size, :]
            v = vis[i * size:(i + 1) * size, j * size:(j + 1) * size, :]
            assert v.shape == (256, 256, 3)

            imsave(img_path + '/' + str(name) + '_' + str(count) + '.png', out)
            imsave(label_path + '/' + str(name) + '_' + str(count) + '.png', gt)
            imsave(vis_path + '/' + str(name) + '_' + str(count) + '.png', v)

            count += 1


def save(img, label, vis, name, save_path, flag='val'):
    img_path = save_path + '/img'
    label_path = save_path + '/label'
    v_path = save_path + '/label_vis'
    check_dir(img_path), check_dir(label_path), check_dir(v_path)

    if flag == 'val':
        crop(img, label, vis, name, img_path, label_path, v_path)
    else:
        crop_overlap(img, label, vis, name, img_path, label_path, v_path)


def run(root_path):
    data_path = root_path + '/dataset_origin/image'
    label_path = root_path + '/dataset_origin/gts'
    label_vis_path = root_path + '/dataset_origin/vis'

    list = os.listdir(label_path)
    for i in list:
        name = i[20:-4]
        print(i, name)

        img = read(os.path.join(data_path, i))
        label = read(os.path.join(label_path, i))
        label_vis = read(os.path.join(label_vis_path, i))

        if int(name) in test_name:
            print(name, 'test')
            test_path = root_path + '/test'
            check_dir(test_path)
            save(img, label, label_vis, name, test_path, flag='test')

        # elif int(name) in val_name:
        #     print(name, 'val')
        #     val_path = root_path + '/val'
        #     check_dir(val_path)
        #     save(img, label, label_vis, name, val_path, flag='val')

        else:
            print(name, 'train')
            train_path = root_path + '/train'
            check_dir(train_path)
            save(img, label, label_vis, name, train_path, flag='train')


if __name__ == '__main__':
    root_path = '/media/hlf/Luffy/WLS/semantic/dataset/vaihingen'
    run(root_path)