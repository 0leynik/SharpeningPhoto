# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from skimage.io import imread, imshow, imsave
import skimage
import keras
import keras.backend as K
import SP_model
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d


def graph_metrics():
    # metrics = np.loadtxt(os.path.expanduser('~/m.csv'), delimiter=',')
    metrics = np.loadtxt('/home/doleinik/SP_metrics.csv', delimiter=',')
    plt.figure('loss')
    plt.plot(metrics[:, 1])
    plt.figure('acc')
    plt.plot(metrics[:, 2])
    plt.show()


def plt_img(data):
    # CxHxW -> HxWxC
    img = np.transpose(data, (1, 2, 0))

    # BGR -> RGB
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[:, :, ::-1]

    # matplotlib.pyplot.imshow()
    # HxWx3 – RGB (float or uint8 array)
    plt.imshow(img)

def evaluate():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    model_path = '/home/doleinik/SP_saved_models/SP_model_iter_37500.h5'
    model = keras.models.load_model(model_path, custom_objects={'laplacian_gray_loss': SP_model.laplacian_gray_loss})

    # from img file
    if True:
        # img_path = '/home/doleinik/SharpeningPhoto/quality_ImageNet/test_500/images/100_fb.JPEG'
        img_path = '/home/doleinik/me.jpg'


        img = skimage.img_as_float(imread(img_path))[:128, :128]
        img = img[:, :, ::-1]  # RGB -> BGR
        img = np.transpose(img, (2, 0, 1))  # HxWxC -> CxHxW

        # img_patches = extract_patches_2d(img, (128, 128))
        # print('shape img_patches: ' + str(img_patches.shape))

        np_img = np.empty((1, 3, 128, 128), dtype=np.float32)
        np_img[0] = img
        predict_img = model.predict(np_img)

        plt.figure('blur')
        plt_img(np_img[0])
        plt.figure('predict')
        plt_img(predict_img[0])

        # sharp_img = skimage.img_as_float(imread('/home/doleinik/SharpeningPhoto/quality_ImageNet/test_500/images/100_sh.JPEG'))[:128, :128]
        # sharp_img = sharp_img[..., ::-1]
        # sharp_img = np.transpose(sharp_img, (2, 0, 1))
        # plt.figure('sharp')
        # plt_img(sharp_img)

        plt.show()

    else:
        # from lmdb
        lmdb_path = '/home/doleinik/SharpeningPhoto/lmdb/'
        # paths = [lmdb_path + 'train_blur_lmdb_128', lmdb_path + 'train_sharp_lmdb_128']
        paths = [lmdb_path + 'val_blur_lmdb_128', lmdb_path + 'val_sharp_lmdb_128']

        id = '{:08}'.format(0)
        train_blur_data, train_sharp_data = SP_model.get_data_from_keys(paths, [id])

        predict_data_1 = model.predict(train_blur_data)
        predict_data_2 = model.predict(predict_data_1)

        plt.figure('blur')
        plt_img(train_blur_data[0])

        plt.figure('sharp')
        plt_img(train_sharp_data[0])

        plt.figure('pred 1')
        plt_img(predict_data_1[0])

        plt.figure('pred 2')
        plt_img(predict_data_2[0])

        plt.show()


if __name__ == '__main__':

    # graph_metrics()
    evaluate()
