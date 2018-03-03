# -*- coding: utf-8 -*-

import keras

from keras.layers import Dense,Input,Conv2D,MaxPooling2D,Conv2DTranspose,Cropping2D,concatenate
from keras.models import Model,Sequential
from keras.optimizers import Adam
import keras.backend as K
import numpy as np

from skimage.io import imread, imshow, imsave
import skimage
import cv2
import matplotlib.pyplot as plt

import SP_model

def plt_img(data):
    # CxHxW -> HxWxC
    img = np.transpose(data, (1, 2, 0))

    # BGR -> RGB
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[:, :, ::-1]

    # matplotlib.pyplot.imshow()
    # HxWx3 – RGB (float or uint8 array)

    img = cv2.cvtColor(np.uint8(img*255), cv2.COLOR_RGB2GRAY)

    plt.imshow(img, cmap='gray')


BGR = [0.114, 0.587, 0.299]

def test_metric():
    img = skimage.img_as_float(imread('/Users/dmitryoleynik/PycharmProjects/SharpeningPhoto/filter_ImageNet_for_visio/4.JPEG')) # RGB
    h, w, c = img.shape
    img = img[..., ::-1]  # BGR
    img = np.transpose(img, (2, 0, 1))
    # print(img.shape)


    img_arr = np.empty((10, 3, h, w))
    img_arr[0] = img

    y_true = K.variable(img_arr)
    y_pred = K.variable(img_arr)


    # y_true_gray = y_true[:, 0:1] * BGR[0] + y_true[:, 1:2] * BGR[1] + y_true[:, 2:3] * BGR[2] # to GRAY
    # y_pred_gray = y_pred[:, 0:1] * BGR[0] + y_pred[:, 1:2] * BGR[1] + y_pred[:, 2:3] * BGR[2]  # to GRAY
    # print(y_true_gray.shape)
    #
    # kernel = K.variable(np.array([[[[-1]], [[-1]], [[-1]]], [[[-1]], [[8]], [[-1]]], [[[-1]], [[-1]], [[-1]]]]), dtype='float32')
    #
    # y_true_conv = K.conv2d(y_true_gray, kernel, (1, 1), 'same', 'channels_first') # edge detection with Laplacian
    # y_true_conv = K.clip(y_true_conv, 0, 1)
    #
    # y_pred_conv = K.conv2d(y_pred_gray, kernel, (1, 1), 'same', 'channels_first') # edge detection with Laplacian
    # y_pred_conv = K.clip(y_pred_conv, 0, 1)
    # print(y_pred_conv.shape)

    kernel = K.variable(np.array([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]] * 3] * 3), dtype='float32')
    print(kernel.shape)
    y_true_conv = K.conv2d(y_true, kernel, (1, 1), 'same', 'channels_first')
    y_true_conv = K.clip(y_true_conv, 0, 1)
    y_pred_conv = K.conv2d(y_pred, kernel, (1, 1), 'same', 'channels_first')
    y_pred_conv = K.clip(y_pred_conv, 0, 1)
    print(y_pred_conv.shape)

    abs = K.abs(y_pred_conv - y_true_conv)
    print(abs.shape)

    mean = K.mean(abs)
    print(mean.shape)

    plt.figure('true')
    plt_img(img)

    plt.figure('pred')
    plt_img(K.eval(y_pred_conv[0]))

    plt.show()

    return mean


def get_super_small_unet():
    # batch 176
    K.set_image_data_format('channels_first')

    img_shape = (3, 375, 500)
    concat_axis = 1

    inputs = Input(shape=img_shape)
    # print(inputs.shape)

    conv1 = Conv2D(3, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(3, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # print(pool1.shape)

    # conv2 = Conv2D(9, (3, 3), activation='relu', padding='same')(pool1)
    # conv2 = Conv2D(9, (3, 3), activation='relu', padding='same')(conv2)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # print(pool2.shape)

    conv3 = Conv2D(9, (3, 3), activation='relu', padding='same')(pool1)
    conv3 = Conv2D(9, (3, 3), activation='relu', padding='same')(conv3)
    deconv = Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='valid')
    crop = Cropping2D(cropping=((0, 0), (1, 0)))
    up3 = deconv(conv3)
    crop_up3 = crop(up3)

    # print(deconv.output_shape, '->', crop.output_shape, conv2.shape)
    # concat4 = concatenate([crop_up3, conv2], axis=concat_axis)
    # conv4 = Conv2D(9, (3, 3), activation='relu', padding='same')(concat4)
    # conv4 = Conv2D(9, (3, 3), activation='relu', padding='same')(conv4)
    # deconv = Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='valid')
    # crop = Cropping2D(cropping=((0, 0), (0, 1)))
    # up4 = deconv(conv4)
    # crop_up4 = crop(up4)

    # print(deconv.output_shape, '->', crop.output_shape, conv1.shape)
    concat5 = concatenate([crop_up3, conv1], axis=concat_axis)
    conv5 = Conv2D(3, (3, 3), activation='relu', padding='same')(concat5)
    conv5 = Conv2D(3, (3, 3), activation='relu', padding='same')(conv5)
    outputs = Conv2D(3, (1, 1), activation='sigmoid')(conv5)
    # print(outputs.shape)

    model = Model(inputs=[inputs], outputs=[outputs])
    # model.compile(optimizer='adam', loss=SP_model.laplacian_gray_loss, metrics=['accuracy'])
    # model.compile(optimizer='adam', loss=SP_model.laplacian_color_loss, metrics=['accuracy'])

    # model.summary()
    # print('Metrics: ' + str(model.metrics_names))
    return model


if __name__ == '__main__':

    test_metric()

    # arr = np.empty((10,3,375,500), dtype=np.float32)
    #
    # model = get_super_small_unet()

    # scores = model.train_on_batch(arr, arr)
    #
    # for m in zip(model.metrics_names,scores):
    #     print(m)
