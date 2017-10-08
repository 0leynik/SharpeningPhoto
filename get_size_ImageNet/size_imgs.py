# -*- coding: utf-8 -*-
import numpy
import glob
import os
import cv2
import matplotlib.pyplot as plt

def get_size_imgs(path, size_arr):

    files = glob.glob(path)
    for f in files:
        if os.path.isfile(f) and f.lower().endswith(".jpeg"):
            # print f
            print(str(len(size_arr)))

            img = cv2.imread(f)
            height, width = img.shape[:2]
            if height >= width:
                size_arr.append(height)
            else:
                size_arr.append(width)
        elif os.path.isdir(f):
            get_size_imgs(f + "/*", size_arr)

if __name__ == "__main__":

    # size_arr = []
    # path = '/home/image-net/ILSVRC2015/Data/CLS-LOC/train/*'
    # get_size_imgs(path, size_arr)
    # numpy.save('train.npy', numpy.array(size_arr))

    size_arr = numpy.load('train.npy')
    plt.hist(size_arr, bins = 1000, histtype = 'stepfilled', color='b')
    plt.show()
