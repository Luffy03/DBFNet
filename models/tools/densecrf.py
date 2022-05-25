import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
import os
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from dataset.datapoint import value_to_rgb
from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian


def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax


def get_crf(opt, mask, img):
    mask = np.transpose(mask, (2, 0, 1))
    img = np.ascontiguousarray(img)

    unary = -np.log(mask + 1e-8)
    unary = unary.reshape((opt.num_classes, -1))
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF2D(opt.img_size, opt.img_size, opt.num_classes)
    d.setUnaryEnergy(unary)

    d.addPairwiseGaussian(sxy=5, compat=3)
    d.addPairwiseBilateral(sxy=10, srgb=13, rgbim=img, compat=10)

    output = d.inference(10)

    map = np.argmax(output, axis=0).reshape((opt.img_size, opt.img_size))

    return map
