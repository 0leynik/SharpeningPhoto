# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
# import lmdb
# import caffe
# from datetime import datetime
import keras

# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Input,Conv2D,MaxPooling2D,Conv2DTranspose,Cropping2D,concatenate
# from keras.models import Model
# from keras.optimizers import Adam
# from keras.losses import mean_squared_error
# from keras import backend as K

import SP_model


def graph_metrics():
    metrics = np.loadtxt(os.path.expanduser('~/m.csv'), delimiter=',')
    plt.figure('loss')
    plt.plot(metrics[:, 0])
    plt.figure('acc')
    plt.plot(metrics[:, 1])
    plt.show()


def plt_img(data):
    # CxHxW -> HxWxC
    img = np.transpose(data, (1, 2, 0))

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # matplotlib.pyplot.imshow()
    # HxWx3 – RGB (float or uint8 array)
    plt.imshow(img)


if __name__ == '__main__':

    model_path = '~/SP_saved_models/SP_model\ iter_num\:39500\ ep\:23\ batch_count\:31808\ loss\:0.00609019\ acc0.207066.h5'
    model = SP_model.get_loaded_model(os.path.expanduser(model_path))

    # img_path = '/home/doleinik/SharpeningPhoto/quality_ImageNet/test_500/images/100_fb.JPEG'

    lmdb_path = '/home/doleinik/SharpeningPhoto/lmdb/'
    train_paths = [lmdb_path + 'train_blur_lmdb_128', lmdb_path + 'train_sharp_lmdb_128']
    train_blur_data, train_sharp_data = SP_model.get_data_from_keys(train_paths, ['{:08}'.format(0)])

    train_blur_data = train_blur_data.astype('float32')
    train_blur_data /= 255
    train_sharp_data = train_sharp_data.astype('float32')
    train_sharp_data /= 255

    predict_data = model.predict(train_blur_data)

    plt.figure('blur')
    plt_img(train_blur_data[0])

    plt.figure('sharp')
    plt_img(train_sharp_data[0])

    plt.figure('pred')
    plt_img(predict_data[0])

    plt.show()
