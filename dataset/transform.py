import cv2
import numpy as np
import os
import math
import collections
import random


class Scale(object):
    """Rescale the input PIL.Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size

    def __call__(self, inputs):
        """
        Args:
            img (npy): Image to be scaled.
        """
        outs = []
        for input in inputs:
            h = w = self.size
            oh, ow, c = input.shape
            img = np.resize(input, (h, w, c))
            outs.append(img)

        return outs


class RandomHorizontalFlip(object):
    def __init__(self, u=0.5):
        self.u = u

    def __call__(self, inputs):
        if np.random.random() < self.u:
            new_inputs = []
            for input in inputs:
                input = np.flip(input, 0)
                new_inputs.append(input)
            return new_inputs
        else:
            return inputs


class RandomVerticleFlip(object):
    def __init__(self, u=0.5):
        self.u = u

    def __call__(self, inputs):
        if np.random.random() < self.u:
            new_inputs = []
            for input in inputs:
                input = np.flip(input, 1)
                new_inputs.append(input)
            return new_inputs
        else:
            return inputs


class RandomRotate90(object):
    def __init__(self, u=0.5):
        self.u = u

    def __call__(self, inputs):
        if np.random.random() < self.u:
            new_inputs = []
            for input in inputs:
                input = np.rot90(input)
                new_inputs.append(input)
            return new_inputs
        else:
            return inputs


class Color_Aug(object):
    def __init__(self):
        self.contra_adj = 0.1
        self.bright_adj = 0.1

    def __call__(self, image):
        n_ch = image.shape[-1]
        ch_mean = np.mean(image, axis=(0, 1), keepdims=True).astype(np.float32)

        contra_mul = np.random.uniform(1 - self.contra_adj, 1 + self.contra_adj, (1, 1, n_ch)).astype(
            np.float32
        )
        bright_mul = np.random.uniform(1 - self.bright_adj, 1 + self.bright_adj, (1, 1, n_ch)).astype(
            np.float32
        )

        image = (image - ch_mean) * contra_mul + ch_mean * bright_mul

        return image


class RandomHueSaturationValue(object):
    def __init__(self):
        self.hue_shift_limit = (-25, 25)
        self.sat_shift_limit = (-15, 15)
        self.val_shift_limit = (-15, 15)

    def change(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(self.hue_shift_limit[0], self.hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(self.val_shift_limit[0], self.val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        # image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image

    def __call__(self, img):
        if np.random.random() < 0.5:
            if img.shape[-1] == 4:
                img_3 = img[:, :, :3]
                img_3 = self.change(img_3)
                img[:, :, :3] = img_3
            else:
                img = self.change(img)

        return img


def stretchImage(data, s=0.005, bins=2000):  # 线性拉伸，去掉最大最小0.5%的像素值，然后线性拉伸至[0,1]
    ht = np.histogram(data, bins)
    d = np.cumsum(ht[0]) / float(data.size)
    lmin = 0
    lmax = bins - 1
    while lmin < bins:
        if d[lmin] >= s:
            break
        lmin += 1
    while lmax >= 0:
        if d[lmax] <= 1 - s:
            break
        lmax -= 1
    return np.clip((data - ht[1][lmin]) / (ht[1][lmax] - ht[1][lmin]), 0, 1)


g_para = {}


def getPara(radius=4):  # 根据半径计算权重参数矩阵
    global g_para
    m = g_para.get(radius, None)
    if m is not None:
        return m
    size = radius * 2 + 1
    m = np.zeros((size, size))
    for h in range(-radius, radius + 1):
        for w in range(-radius, radius + 1):
            if h == 0 and w == 0:
                continue
            m[radius + h, radius + w] = 1.0 / math.sqrt(h ** 2 + w ** 2)
    m /= m.sum()
    g_para[radius] = m
    return m


def zmIce(I, ratio=4, radius=300):  # 常规的ACE实现
    para = getPara(radius)
    height, width = I.shape
    #    zh,zw = [0]*radius + range(height) + [height-1]*radius, [0]*radius + range(width)  + [width -1]*radius
    zh, zw = [0] * radius + [x for x in range(height)] + [height - 1] * radius, [0] * radius + [x for x in
                                                                                                range(width)] + [
                 width - 1] * radius
    Z = I[np.ix_(zh, zw)]
    res = np.zeros(I.shape)
    for h in range(radius * 2 + 1):
        for w in range(radius * 2 + 1):
            if para[h][w] == 0:
                continue
            res += (para[h][w] * np.clip((I - Z[h:h + height, w:w + width]) * ratio, -1, 1))
    return res


def zmIceFast(I, ratio=4, radius=300):  # 单通道ACE快速增强实现
    height, width = I.shape[:2]
    if min(height, width) <= 2:
        return np.zeros(I.shape) + 0.5
    Rs = cv2.resize(I, ((width + 1) // 2, (height + 1) // 2))
    Rf = zmIceFast(Rs, ratio, radius)  # 递归调用
    Rf = cv2.resize(Rf, (width, height))
    Rs = cv2.resize(Rs, (width, height))

    return Rf + zmIce(I, ratio, radius) - zmIce(Rs, ratio, radius)


def zmIceColor(I, ratio=4, radius=3):  # rgb三通道分别增强，ratio是对比度增强因子，radius是卷积模板半径
    res = np.zeros(I.shape)
    for k in range(3):
        res[:, :, k] = stretchImage(zmIceFast(I[:, :, k], ratio, radius))
    return res


def do_gamma(image, gamma=1.0):
    image = image ** (1.0 / gamma)
    image = np.clip(image, 0, 1)
    return image