import os
import numpy as np
import cv2


def ask_for_query(index, path):
    index = list(index)
    query_all = []

    path_list = os.listdir(path)

    for idx in index:
        query = []
        for path_index in path_list:
            if (set(list(path_index)) & set(index)) == set(idx):
                query.append(path_index)
        query_all.append((idx, query))

    query_dict = dict(query_all)

    return query_dict


def ask_for_query_new(index, path):
    index = list(index)
    query_all = []

    path_list = os.listdir(path)

    for i in path_list:
        # the shared
        I = (set(list(i)) & set(index))
        if I != set():
            if len(index) > 1:
                if I != set(index):
                    query_all.append(i)
            else:
                query_all.append(i)

    return query_all


if __name__ == '__main__':
    path = '/home/ggm/WLS/semantic/dataset/potsdam/train/img_index'

    index = '012'
    query = ask_for_query_new(index, path)
    print(query)

