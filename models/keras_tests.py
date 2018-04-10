# -*- coding: utf-8 -*-

import keras

from keras.layers import Dense,Input,Conv2D,MaxPooling2D,Conv2DTranspose,Cropping2D,concatenate,Concatenate
from keras.models import Model,Sequential
from keras.optimizers import Adam
import keras.backend as K
import numpy as np

from skimage.io import imread, imshow, imsave
import skimage
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib as mpl


# import SP_model

def plt_img(data):
    # CxHxW -> HxWxC
    img = np.transpose(data, (1, 2, 0))

    # BGR -> RGB
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[:, :, ::-1]


    # img = cv2.cvtColor(np.uint8(img*255), cv2.COLOR_RGB2GRAY)
    # plt.imshow(img, cmap='gray')

    print(np.min(img),np.max(img))

    img = np.clip(img, 0, 1)

    # matplotlib.pyplot.imshow()
    # HxWx3 – RGB (float or uint8 array)
    # plt.imshow(img[..., 0], cmap='Reds')

    # plt.imshow(img[...,2], cmap='gray')
    plt.imshow(img)


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

    y_pred = K.variable(img_arr) # предсказано размытое фото
    gaus_kernel = K.variable(np.array([[[[1]], [[2]], [[1]]], [[[2]], [[4]], [[2]]], [[[1]], [[2]], [[1]]]]), dtype='float32') / 16
    b_pred = K.conv2d(y_pred[:, 0:1], gaus_kernel, (1, 1), 'same', 'channels_first')
    g_pred = K.conv2d(y_pred[:, 1:2], gaus_kernel, (1, 1), 'same', 'channels_first')
    r_pred = K.conv2d(y_pred[:, 2:3], gaus_kernel, (1, 1), 'same', 'channels_first')
    y_pred = K.concatenate([b_pred, g_pred, r_pred], axis=1)
    y_pred = K.clip(y_pred, 0, 1)


    # y_true_gray = y_true[:, 0:1] * BGR[0] + y_true[:, 1:2] * BGR[1] + y_true[:, 2:3] * BGR[2] # to GRAY
    # y_pred_gray = y_pred[:, 0:1] * BGR[0] + y_pred[:, 1:2] * BGR[1] + y_pred[:, 2:3] * BGR[2]  # to GRAY
    # print(y_true_gray.shape)
    #
    # kernel = K.variable(np.array([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]] * 3] * 3), dtype='float32')
    # kernel = K.variable(np.array([[[[-1]], [[-1]], [[-1]]], [[[-1]], [[8]], [[-1]]], [[[-1]], [[-1]], [[-1]]]]), dtype='float32')
    #
    # y_true_conv = K.conv2d(y_true_gray, kernel, (1, 1), 'same', 'channels_first') # edge detection with Laplacian
    # y_true_conv = K.clip(y_true_conv, 0, 1)
    #
    # y_pred_conv = K.conv2d(y_pred_gray, kernel, (1, 1), 'same', 'channels_first') # edge detection with Laplacian
    # y_pred_conv = K.clip(y_pred_conv, 0, 1)
    # print(y_pred_conv.shape)


    kernel = K.variable(np.array([[[[-1]], [[-1]], [[-1]]], [[[-1]], [[8]], [[-1]]], [[[-1]], [[-1]], [[-1]]]]), dtype='float32')
    print(kernel.shape)

    b_true = K.conv2d(y_true[:, 0:1], kernel, (1, 1), 'same', 'channels_first')
    g_true = K.conv2d(y_true[:, 1:2], kernel, (1, 1), 'same', 'channels_first')
    r_true = K.conv2d(y_true[:, 2:3], kernel, (1, 1), 'same', 'channels_first')
    y_true_conv = K.concatenate([b_true,g_true,r_true], axis=1)
    # y_true_conv = K.clip(y_true_conv, 0, 1)

    b_pred = K.conv2d(y_pred[:, 0:1], kernel, (1, 1), 'same', 'channels_first')
    g_pred = K.conv2d(y_pred[:, 1:2], kernel, (1, 1), 'same', 'channels_first')
    r_pred = K.conv2d(y_pred[:, 2:3], kernel, (1, 1), 'same', 'channels_first')
    y_pred_conv = K.concatenate([b_pred,g_pred,r_pred], axis=1)
    # y_pred_conv = K.clip(y_pred_conv, 0, 1)
    print(y_pred_conv.shape)

    clip = K.clip((y_true_conv - y_pred_conv), 0, 1)
    diff = y_true_conv - y_pred_conv

    # plt.figure('true')
    # plt_img(K.eval(y_true)[0])
    #
    # plt.figure('true conv')
    # plt_img(K.eval(y_true_conv)[0])
    #
    # plt.figure('pred')
    # plt_img(K.eval(y_pred)[0])
    #
    # plt.figure('pred conv')
    # plt_img(K.eval(y_pred_conv)[0])

    plt.figure('clip')
    # plt_img(K.eval(clip)[0])
    plt_img(K.eval(clip)[0])

    plt.figure('diff')
    plt_img(K.eval(clip)[0])
    # x = K.eval(diff)[0]
    # plt_img((x-np.min(x))/(np.max(x)-np.min(x)))

    plt.show()

    # return mean
    return 1


def mean_squared_error(y_true, y_pred):
    print(y_pred.shape)
    print(y_true.shape)

    mean_1 = K.mean(K.square(y_pred - y_true), axis=-2)
    return mean_1

    mean = K.mean(mean_1)

    return mean



def get_super_small_unet():
    # batch 176
    K.set_image_data_format('channels_first')

    img_shape = (3, 375, 500)
    concat_axis = 1

    inputs = Input(shape=img_shape)
    # print(inputs.shape)

    conv1 = Conv2D(3, (3, 3), activation='relu', padding='same',kernel_initializer=keras.initializers.glorot_uniform(100))(inputs)
    conv1 = Conv2D(3, (3, 3), activation='relu', padding='same',kernel_initializer=keras.initializers.glorot_uniform(100))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # print(pool1.shape)

    # conv2 = Conv2D(9, (3, 3), activation='relu', padding='same')(pool1)
    # conv2 = Conv2D(9, (3, 3), activation='relu', padding='same')(conv2)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # print(pool2.shape)

    conv3 = Conv2D(9, (3, 3), activation='relu', padding='same',kernel_initializer=keras.initializers.glorot_uniform(100))(pool1)
    conv3 = Conv2D(9, (3, 3), activation='relu', padding='same',kernel_initializer=keras.initializers.glorot_uniform(100))(conv3)
    deconv = Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='valid',kernel_initializer=keras.initializers.glorot_uniform(100))
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
    conv5 = Conv2D(3, (3, 3), activation='relu', padding='same',kernel_initializer=keras.initializers.glorot_uniform(100))(concat5)
    conv5 = Conv2D(3, (3, 3), activation='relu', padding='same',kernel_initializer=keras.initializers.glorot_uniform(100))(conv5)
    outputs = Conv2D(3, (1, 1), activation='sigmoid',kernel_initializer=keras.initializers.glorot_uniform(1000))(conv5)
    # print(outputs.shape)

    model = Model(inputs=[inputs], outputs=[outputs])
    # model.compile(optimizer='adam', loss=SP_model.laplacian_gray_loss, metrics=['accuracy'])
    # model.compile(optimizer='adam', loss=SP_model.abs_laplacian_color_loss, metrics=['accuracy'])
    # model.compile(optimizer='adam', loss=keras.losses.mean_squared_error, metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.compile(optimizer='sgd', loss=mean_squared_error, metrics=['accuracy'])

    # model.summary()
    # print('Metrics: ' + str(model.metrics_names))
    return model


def conv2d_upsampling():

    K.set_image_data_format('channels_first')
    img_shape = (3, 128, 128)
    concat_axis = 1

    inputs = Input(shape=img_shape)

    # 1-line
    conv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    conv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1_1)
    print(conv1_1.shape)
    deconv = Conv2DTranspose(32, (5, 5))
    deconv1_1 = deconv(conv1_1)
    print(deconv.output_shape)

    conv2_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(deconv1_1)
    conv2_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv2_1)
    pool2_1 = MaxPooling2D(pool_size=(2, 2))(conv2_1)

    # concat3_1 = Concatenate(axis=concat_axis)([conv1_1, pool2_1])
    # conv3_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(concat3_1)
    # conv3_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3_1)
    #
    # # 2-line
    # conv1_2 = Conv2D(16, (5, 5), activation='relu', padding='same')(inputs)
    # conv1_2 = Conv2D(16, (5, 5), activation='relu', padding='same')(conv1_2)
    # deconv1_2 = Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same')(conv1_2)
    #
    # conv2_2 = Conv2D(16, (5, 5), activation='relu', padding='same')(deconv1_2)
    # conv2_2 = Conv2D(16, (5, 5), activation='relu', padding='same')(conv2_2)
    # pool2_2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)


    if False:
        # img = skimage.img_as_float(imread('/Users/dmitryoleynik/PycharmProjects/SharpeningPhoto/filter_ImageNet_for_visio/4_3.JPEG')) # RGB
        img = skimage.img_as_float(imread('4_3.JPG')) # RGB
        h, w, c = img.shape
        # img = img[..., ::-1]  # BGR
        # img = np.transpose(img, (2, 0, 1))
        # print(img.shape)
        # plt.figure('img', (16,9))

        print(mpl.rcParams['figure.figsize'])
        print(mpl.rcParams['figure.dpi'])


        plt.figure('img')
        plt.plot(np.random.rand(1000))
        plt.savefig('kek.png')
        plt.show()


if __name__ == '__main__':

    conv2d_upsampling()

    # test_metric()

    # rs = np.random.RandomState(1234)
    # training = rs.rand(10,3,375,500)
    #
    # rs = np.random.RandomState(4321)
    # target = rs.rand(10,3,375,500)
    #
    # print(training.dtype)
    # print(training[1,1,1,1])
    # print(target.dtype)
    # print(target[1,1,1,1])
    #
    # print(K.eval(K.mean(K.variable(training))))
    #
    # # arr = np.empty((10,3,375,500), dtype=np.float32)
    # # arr = np.zeros((10,3,375,500), dtype=np.float32)
    #
    # model = get_super_small_unet()
    # #
    # scores = model.train_on_batch(training, target)
    #
    # for m in zip(model.metrics_names,scores):
    #     print(m)
