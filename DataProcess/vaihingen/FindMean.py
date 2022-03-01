import os
import numpy as np
import tifffile

# means = [119.14901543,  83.04203606,  81.79810095]
path = '/home/ggm/WLS/semantic/dataset/vaihingen/train/img'

list = os.listdir(path)

k = 0
sum = np.zeros([3])

for idx, i in enumerate(list):
    print(idx)
    k += 1
    img = tifffile.imread(os.path.join(path, i))
    img = img.reshape(256*256, -1)

    mean = np.mean(img, axis=0)
    sum += mean


means = sum/k
print(means)

# std = [55.63038161, 40.67145608, 38.61447761]
# means = [119.14901543,  83.04203606,  81.79810095]
#
# path = '/home/ggm/WLS/semantic/dataset/vaihingen/train/img'
#
# list = os.listdir(path)
# k = 0
# sum = np.zeros([3])
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