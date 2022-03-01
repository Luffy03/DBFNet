import os
import random


def write_train(path):
    train_txt = open(path + '/datafiles/potsdam/seg_train.txt', 'w')
    train_path = '/media/hlf/Luffy/WLS//semantic/dataset/potsdam/train/img'
    list = os.listdir(train_path)

    for idx, i in enumerate(list):
        train_txt.write(i + '\n')
    train_txt.close()


def write_val(path):
    val_txt = open(path + '/datafiles/potsdam/seg_val.txt', 'w')
    val_path = '/media/hlf/Luffy/WLS//semantic/dataset/potsdam/val/img'
    list = os.listdir(val_path)

    for idx, i in enumerate(list):
        val_txt.write(i + '\n')
    val_txt.close()


def write_test(path):
    test_txt = open(path + '/datafiles/potsdam/seg_test.txt', 'w')
    test_path = '/media/hlf/Luffy/WLS//semantic/dataset/potsdam/test/img'
    list = os.listdir(test_path)

    for idx, i in enumerate(list):
        test_txt.write(i + '\n')
    test_txt.close()


if __name__ == '__main__':
    path = '/media/hlf/Luffy/WLS//PointAnno'
    write_train(path)
    write_val(path)
    write_test(path)