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

            if height > width:
                w.append(height)
                h.append(width)
            else:
                w.append(width)
                h.append(height)

        elif os.path.isdir(f):
            get_size_imgs(f + "/*", w, h)


if __name__ == "__main__":

    w = []
    h = []
    path = '/home/doleinik/SharpeningPhoto/quality_ImageNet/train_500/images/*'
    get_size_imgs(path, w, h)

    path = '/home/doleinik/SharpeningPhoto/quality_ImageNet/test_500/images/*'
    get_size_imgs(path, w, h)

    path = '/home/doleinik/SharpeningPhoto/quality_ImageNet/val_500/images/*'
    get_size_imgs(path, w, h)

    numpy.save('w.npy', numpy.array(w))
    numpy.save('h.npy', numpy.array(h))


    # w = numpy.load('w.npy')
    # h = numpy.load('h.npy')
    #
    # print('w_mean = ' + str( w.mean() ))
    # print('h_mean = ' + str( h.mean() ))
    #
    # plt.figure('w')
    # plt.hist(w, bins = 1000, histtype = 'stepfilled', color='b')
    # plt.figure('h')
    # plt.hist(h, bins = 1000, histtype = 'stepfilled', color='b')
    # plt.show()
