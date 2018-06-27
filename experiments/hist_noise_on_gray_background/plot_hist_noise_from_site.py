# -*- coding: utf-8 -*-
import numpy as np
import glob
import os
import cv2
import matplotlib.pyplot as plt
import math


if __name__ == '__main__':
    '''
    Свои гистограммы шума с примеров на сайте http://www.cambridgeincolour.com/ru/tutorials-ru/image-noise-2.htm
    Гистограммы для 5 фоток:
    1 шт - серое,
    2 шт - зависимость от пространственной частоты
    2 шт - зависимость от амплитуды
    '''
    files = filter(lambda f: f.endswith('.jpg'), os.listdir('.'))
    print files

    bins = range(0, 256)

    for f in files:

        path_to_save, basename = os.path.split(f)
        # name, ext = os.path.splitext(os.path.basename(f))
        name, ext = os.path.splitext(basename)

        img = cv2.imread(f)
        gray = cv2.imread(f, 0)

        print '\n'+basename
        print 'RGB'
        print 'mean = ' + str(img.mean())
        print 'std = ' + str(img.std())

        plt.figure('RBG '+f,(5,4))
        plt.grid(True, axis='x', linestyle=':')

        x = range(0, 256, 16)
        # x.append(int(img.mean() - img.std()))
        # x.append(int(img.mean() + img.std()))
        plt.xticks(x) #значения на оси


        plt.hist(img[:,:,0].ravel(), bins, histtype = 'stepfilled', color='b', edgecolor='b', alpha = 0.5)
        plt.hist(img[:,:,1].ravel(), bins, histtype = 'stepfilled', color='g', edgecolor='g', alpha = 0.5)
        plt.hist(img[:,:,2].ravel(), bins, histtype = 'stepfilled', color='r', edgecolor='r', alpha = 0.5)
        plt.hist(img.ravel(), bins, histtype = 'stepfilled', color='k', edgecolor='k', alpha = 0.5)
        # plt.hist(img.ravel(), bins, histtype = 'bar', color='k', edgecolor='r')
        # plt.figure('GRAY '+f)
        # plt.hist(gray.ravel(), bins, histtype='bar',edgecolor='yellow' )
        plt.xlim(80, 176) #интервал оси
        plt.ylim(0, 1300) #интервал оси

    plt.show()