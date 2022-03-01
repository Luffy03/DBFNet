import os
import numpy as np
import tifffile

# means = [86.42521457, 92.37607528, 85.74658389, 98.17242502]
path = '/home/ggm/WLS/semantic/dataset/potsdam/train/img'

list = os.listdir(path)

k = 0
sum = np.zeros([4])

for idx, i in enumerate(list):
    print(idx)
    k += 1
    img = tifffile.imread(os.path.join(path, i))
    img = img.reshape(256*256, -1)

    mean = np.mean(img, axis=0)
    sum += mean


means = sum/k
print(means)

# std = [35.58409409, 35.45218542, 36.91464009, 36.3449891]
# means = [86.42521457, 92.37607528, 85.74658389, 98.17242502]
#
# path = '/home/ggm/WLS/semantic/dataset/potsdam/train/img'
#
# list = os.listdir(path)
# k = 0
# sum = np.zeros([4])
#
# for idx, i in enumerate(list):
#     print(idx)
#     k += 1
#     img = tifffile.imread(os.path.join(path, i))
#     img = img.reshape(256*256, -1)
#
#     x = (img - means) ** 2
#
#     sum += np.sum(x, axis=0)
#
# std = np.sqrt(sum/(k*256*256))
# print(std)