# -*- coding: utf-8 -*-

from __future__ import print_function

import os
if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.io import imread, imshow, imsave
import skimage
import keras
import keras.backend as K
import SP_model
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d


iter_num = str(39500);
train_name = 'mean_squared_error_lr_0.001'

def graph_metrics():
    model_path = '/home/doleinik/trained_models_SharpeningPhoto/' + train_name + '/SP_metrics.csv'

    metrics = np.loadtxt(model_path, delimiter=',')
    # metrics = np.loadtxt(os.path.expanduser('~/m.csv'), delimiter=',')

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

h_step, w_step = (128, 128)
h_count = 0
w_count = 0

def get_patches_from_img(img):
    global h_count, w_count
    h, w, c = img.shape

    h_count = h / h_step
    if (h % h_step) != 0:
        h_count = h / h_step + 1
    h_new = h_count * h_step

    w_count = w / w_step
    if (w % w_step) != 0:
        w_count = w / w_step + 1
    w_new = w_count * w_step

    img_new = np.zeros((h_new, w_new, c), dtype=np.float32)
    img_new[:h, :w] = img

    i = 0
    patches = np.zeros((h_count * w_count, h_step, w_step, c), dtype=np.float32)

    for height in np.split(img_new, h_count, axis=0):
        print(height.shape)
        for width in np.split(height, w_count, axis=1):
            print(width.shape)
            patches[i] = width
            i += 1

    return patches

def get_img_from_patches(patches, img):
    h, w, c = img.shape
    b_count = patches.shape[0]
    print(patches.shape)
    extended_img = np.zeros((h_count * h_step, w_count * w_step, c), dtype=np.float32)
    k = 0
    for i in range(h_count):
        for j in range(w_count):
            extended_img[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step] = patches[k]
            k += 1

    return extended_img[:h, :w]


def evaluate():

    # cluster run
    model_path = '/home/doleinik/trained_models_SharpeningPhoto/'+train_name+'/SP_saved_models/SP_model_iter_'+iter_num+'.h5'

    # loacal
    # model_path = '/Users/dmitryoleynik/PycharmProjects/SharpeningPhoto/models/SP_model_iter_39500_mse.h5'
    # model_path = '/Users/dmitryoleynik/PycharmProjects/SharpeningPhoto/models/SP_model_iter_39500_sub_loss.h5'

    # load model
    custom_objects = {'laplacian_gray_loss': SP_model.laplacian_gray_loss}
    model = keras.models.load_model(model_path, custom_objects=custom_objects)


    # execute for single image file
    if True:
        # img_path = '/home/doleinik/SharpeningPhoto/quality_ImageNet/test_500/images/100_fb.JPEG'
        # img_path = '/home/doleinik/me.jpg'
        img_path = '9.JPG'

        original_img = skimage.img_as_float(imread(img_path))
        # original_img = skimage.img_as_float(imread(img_path))[:128,:128]
        print(original_img.shape)
        img = original_img[:, :, ::-1]  # RGB -> BGR
        img_patches = get_patches_from_img(img)
        img_patches = np.array(map(lambda i: np.transpose(i, (2, 0, 1)), img_patches), np.float32)  # HxWxC -> CxHxW
        print('shape img_patches: ' + str(img_patches.shape))

        # np_img = np.empty((1, 3, 128, 128), dtype=np.float32)
        # np_img[0] = np.transpose(img, (2, 0, 1))
        print('Predicting...')
        # predict_img = model.predict(np_img)
        predict_img_patches = model.predict(img_patches)

        predict_img_patches = np.array(map(lambda i: np.transpose(i, (1, 2, 0)), predict_img_patches), np.float32)  # HxWxC -> CxHxW
        predict_img = get_img_from_patches(predict_img_patches, original_img)
        predict_img = predict_img[:, :, ::-1]  # BGR -> RGB

        print('Plotting...')

        plt.figure('blur')
        imsave('blur.JPG', original_img)
        # plt.imshow(original_img)

        plt.figure('predict')
        imsave('predict.JPG', predict_img)
        # plt.imshow(predict_img)
        # plt.imshow(predict_img_patches[0][..., ::-1])

        # sharp_img = skimage.img_as_float(imread('/home/doleinik/SharpeningPhoto/quality_ImageNet/test_500/images/100_sh.JPEG'))[:128, :128]
        # sharp_img = sharp_img[..., ::-1]
        # sharp_img = np.transpose(sharp_img, (2, 0, 1))
        # plt.figure('sharp')
        # plt_img(sharp_img)

        # plt.show()

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

    graph_metrics()
    # evaluate()
