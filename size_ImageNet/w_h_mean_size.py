# -*- coding: utf-8 -*-
import numpy
import glob
import os
import cv2
import matplotlib.pyplot as plt

def get_size_imgs(path, w, h):

    files = glob.glob(path)
    for f in files:
        if os.path.isfile(f) and f.lower().endswith(".jpeg"):
            print f

            img = cv2.imread(f)
            height, width = img.shape[:2]
            h.append(height)
            w.append(width)

        elif os.path.isdir(f):
            get_size_imgs(f + "/*", w, h)


if __name__ == "__main__":

    w = []
    h = []
    path = 'train_500/images/*'
    get_size_imgs(path, w, h)

    path = 'test_500/images/*'
    get_size_imgs(path, w, h)

    path = 'val_500/images/*'
    get_size_imgs(path, w, h)

    print('w_mean = ' + str( numpy.array(w).mean() ))
    print('h_mean = ' + str( numpy.array(h).mean() ))