# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import cv2
import lmdb

import pkgutil
found_caffe = pkgutil.find_loader('caffe') is not None
if found_caffe:
    import caffe
print('Caffe found = '+str(found_caffe))

from datetime import datetime

import keras
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Input,Conv2D,MaxPooling2D,Conv2DTranspose,Cropping2D, Concatenate, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.initializers import glorot_normal
from keras import backend as K

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
limit_mem = False
if limit_mem:
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.Session(config=config))



def gen_batch_keylists(N, batch_size):
    '''
    формирование батчей в виде списков ключей в формате '{:08}'
    :return [['00000000','00000001',...],['','',...],['','',...]]
    '''

    keys = np.arange(N)
    np.random.shuffle(keys)
    keys = map(lambda key: '{:08}'.format(key), keys)

    batches = []
    k = 1;
    while k*batch_size < N:
        batches.append( keys[(k-1)*batch_size : k*batch_size] )
        k += 1
    if k*batch_size >= N:
        batches.append( keys[(k-1)*batch_size : N] )

    return batches


def get_data_from_keys(lmdb_paths, keylist):
    '''
    формирование списка из np-данных на input и loss
    :param lmdb_paths: [blur_lmdb_path, sharp_lmdb_path]
    :param keylist: список ключей в формате '{:08}'
    :return: [np_blur_data, np_sharp_data]
    '''

    # print(str(datetime.now())+'    Load data...')

    visualize = False
    batch_size = len(keylist)

    blur_data = np.empty((batch_size, 3, IMG_H, IMG_W), dtype=np.uint8)
    sharp_data = np.empty((batch_size, 3, IMG_H, IMG_W), dtype=np.uint8)
    ret_data = [blur_data, sharp_data]

    if found_caffe:
        datum = caffe.proto.caffe_pb2.Datum()
        for i in range(2):

            env = lmdb.open(lmdb_paths[i], readonly=True)
            txn = env.begin() # можно делать get() из txn
            # curs = txn.cursor() # можно делать get() из txn.cursor
            # value = curs.get(key)

            # Conv2D
            # data_format: channels_first
            # shape(batch, channels, height, width)

            for j in range(batch_size):
                value = txn.get(keylist[j])
                datum.ParseFromString(value)
                data = caffe.io.datum_to_array(datum) # (datum.channels, datum.height, datum.width)
                ret_data[i][j] = data[:, :IMG_H, :IMG_W]

                # print(type(data))
                # print(data.dtype)
                # print(data.shape)
                if visualize:
                    # CxHxW -> HxWxC
                    img = np.transpose(data, (1, 2, 0))
                    # BGR -> RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # matplotlib.pyplot.imshow()
                    # HxWx3 – RGB (float or uint8 array)
                    plt.imshow(img)
                    plt.show()

    # print('Batch size in memory = ' + str(1. * ret_data[0].nbytes / (pow(2, 30))) + ' GB')
    for i in range(len(ret_data)):
        ret_data[i] = ret_data[i].astype('float32')
        ret_data[i] /= 255
    # print('Batch size in memory = ' + str(1. * ret_data[0].nbytes / (pow(2, 30))) + ' GB')

    return ret_data


BGR = [0.114, 0.587, 0.299]
kernel = K.variable(np.array([[[[-1]], [[-1]], [[-1]]], [[[-1]], [[8]], [[-1]]], [[[-1]], [[-1]], [[-1]]]]), dtype='float32')
# print(kernel.shape)

def laplacian_gray_loss(y_true, y_pred):

    y_true_gray = y_true[:, 0:1] * BGR[0] + y_true[:, 1:2] * BGR[1] + y_true[:, 2:3] * BGR[2]  # to GRAY
    y_pred_gray = y_pred[:, 0:1] * BGR[0] + y_pred[:, 1:2] * BGR[1] + y_pred[:, 2:3] * BGR[2]  # to GRAY
    # print(y_true_gray.shape)

    y_true_conv = K.conv2d(y_true_gray, kernel, (1, 1), 'same', 'channels_first')  # edge detection with Laplacian
    y_true_conv = K.clip(y_true_conv, 0, 1)

    y_pred_conv = K.conv2d(y_pred_gray, kernel, (1, 1), 'same', 'channels_first')  # edge detection with Laplacian
    y_pred_conv = K.clip(y_pred_conv, 0, 1)
    # print(y_pred_conv.shape)

    abs = K.abs(y_pred_conv - y_true_conv) # тоже были перепутаны y_pred_conv и y_true_conv, если юзать - поменять местами
    # print(abs.shape)

    mean = K.mean(abs)
    # print(mean.shape)

    return mean

def clip_laplacian_color_loss(y_true, y_pred):

    b_true = K.conv2d(y_true[:, 0:1], kernel, (1, 1), 'same', 'channels_first')
    g_true = K.conv2d(y_true[:, 1:2], kernel, (1, 1), 'same', 'channels_first')
    r_true = K.conv2d(y_true[:, 2:3], kernel, (1, 1), 'same', 'channels_first')
    y_true_conv = K.concatenate([b_true, g_true, r_true], axis=1)
    # y_true_conv = K.clip(y_true_conv, 0, 1) # убрано чтобы учитывать разницу лучше, clip может обрезать данные

    b_pred = K.conv2d(y_pred[:, 0:1], kernel, (1, 1), 'same', 'channels_first')
    g_pred = K.conv2d(y_pred[:, 1:2], kernel, (1, 1), 'same', 'channels_first')
    r_pred = K.conv2d(y_pred[:, 2:3], kernel, (1, 1), 'same', 'channels_first')
    y_pred_conv = K.concatenate([b_pred, g_pred, r_pred], axis=1)
    # y_pred_conv = K.clip(y_pred_conv, 0, 1) # убрано чтобы учитывать разницу лучше, clip может обрезать данные
    # print(y_pred_conv.shape)

    clip = K.clip((y_true_conv - y_pred_conv), 0, 1)
    return clip
    # diff = y_true_conv - y_pred_conv
    # return diff

def sub_loss(y_true, y_pred):
    return K.square(y_true - y_pred)


# def get_unet():
#     # batch 170
#
#     img_shape = (3, IMG_H, IMG_W)
#     concat_axis = 1
#
#     inputs = Input(shape=img_shape)
#     print(inputs.shape)
#
#     conv1 = Conv2D(2, (3, 3), activation='relu', padding='same')(inputs)
#     conv1 = Conv2D(2, (3, 3), activation='relu', padding='same')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#     print(pool1.shape)
#
#     conv2 = Conv2D(4, (3, 3), activation='relu', padding='same')(pool1)
#     conv2 = Conv2D(4, (3, 3), activation='relu', padding='same')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#     print(pool2.shape)
#
#     conv3 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool2)
#     conv3 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#     print(pool3.shape)
#
#     conv4 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool3)
#     conv4 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
#     print(pool4.shape)
#
#     conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool4)
#     conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
#     deconv = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')
#     up5 = deconv(conv5)
#
#     print(deconv.output_shape, conv4.shape)
#     # crop4 = Cropping2D(cropping=((1,0),(1,0)))(conv4)
#     concat6 = concatenate([up5, conv4], axis=concat_axis)
#     conv6 = Conv2D(16, (3, 3), activation='relu', padding='same')(concat6)
#     conv6 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv6)
#     deconv = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='valid')
#     up6 = deconv(conv6)
#
#     print(deconv.output_shape, conv3.shape)
#     # crop3 = Cropping2D(cropping=((1,0),(1,0)))(conv3)
#     # concat7 = concatenate([up6, crop3], axis=concat_axis)
#     concat7 = concatenate([up6, conv3], axis=concat_axis)
#     conv7 = Conv2D(8, (3, 3), activation='relu', padding='same')(concat7)
#     conv7 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv7)
#     deconv = Conv2DTranspose(4, (3, 3), strides=(2, 2), padding='valid')
#     crop = Cropping2D(cropping=((0, 0), (1, 0)))
#     up7 = deconv(conv7)
#     crop_up7 = crop(up7)
#
#     print(deconv.output_shape, '->', crop.output_shape, conv2.shape)
#     # crop2 = Cropping2D(cropping=((1, 0), (0, 0)))(conv2)
#     # concat8 = concatenate([up7, crop2], axis=concat_axis)
#     concat8 = concatenate([crop_up7, conv2], axis=concat_axis)
#     conv8 = Conv2D(4, (3, 3), activation='relu', padding='same')(concat8)
#     conv8 = Conv2D(4, (3, 3), activation='relu', padding='same')(conv8)
#     deconv = Conv2DTranspose(2, (3, 3), strides=(2, 2), padding='valid')
#     crop = Cropping2D(cropping=((0, 0), (0, 1)))
#     up8 = deconv(conv8)
#     crop_up8 = crop(up8)
#
#     print(deconv.output_shape, '->', crop.output_shape, conv1.shape)
#     # crop1 = Cropping2D(cropping=((0, 0), (0, 0)))(conv1)
#     # concat9 = concatenate([up8, crop1], axis=concat_axis)
#     concat9 = concatenate([crop_up8, conv1], axis=concat_axis)
#     conv9 = Conv2D(2, (3, 3), activation='relu', padding='same')(concat9)
#     conv9 = Conv2D(2, (3, 3), activation='relu', padding='same')(conv9)
#     outputs = Conv2D(3, (1, 1), activation='sigmoid')(conv9)
#     print(outputs.shape)
#
#     model = Model(inputs=[inputs], outputs=[outputs])
#
#     # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=['accuracy'])
#     # model.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=[dice_coef])
#     # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
#
#     # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mse', dice_coef])
#     # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse', dice_coef])
#     # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mse', dice_coef])
#     model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#
#     model.summary()
#     print('Metrics: ' + str(model.metrics_names))
#     return model
#
# def get_small_unet():
#     # batch 176
#
#     img_shape = (3, IMG_H, IMG_W)
#     concat_axis = 1
#
#     inputs = Input(shape=img_shape)
#     print(inputs.shape)
#
#     conv1 = Conv2D(3, (3, 3), activation='relu', padding='same')(inputs)
#     conv1 = Conv2D(3, (3, 3), activation='relu', padding='same')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#     print(pool1.shape)
#
#     conv2 = Conv2D(9, (3, 3), activation='relu', padding='same')(pool1)
#     conv2 = Conv2D(9, (3, 3), activation='relu', padding='same')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#     print(pool2.shape)
#
#     conv3 = Conv2D(18, (3, 3), activation='relu', padding='same')(pool2)
#     conv3 = Conv2D(18, (3, 3), activation='relu', padding='same')(conv3)
#     deconv = Conv2DTranspose(9, (3, 3), strides=(2, 2), padding='valid')
#     crop = Cropping2D(cropping=((0, 0), (1, 0)))
#     up3 = deconv(conv3)
#     crop_up3 = crop(up3)
#
#     print(deconv.output_shape, '->', crop.output_shape, conv2.shape)
#     concat4 = concatenate([crop_up3, conv2], axis=concat_axis)
#     conv4 = Conv2D(9, (3, 3), activation='relu', padding='same')(concat4)
#     conv4 = Conv2D(9, (3, 3), activation='relu', padding='same')(conv4)
#     deconv = Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='valid')
#     crop = Cropping2D(cropping=((0, 0), (0, 1)))
#     up4 = deconv(conv4)
#     crop_up4 = crop(up4)
#
#     print(deconv.output_shape, '->', crop.output_shape, conv1.shape)
#     concat5 = concatenate([crop_up4, conv1], axis=concat_axis)
#     conv5 = Conv2D(3, (3, 3), activation='relu', padding='same')(concat5)
#     conv5 = Conv2D(3, (3, 3), activation='relu', padding='same')(conv5)
#     outputs = Conv2D(3, (1, 1), activation='sigmoid')(conv5)
#     print(outputs.shape)
#
#     model = Model(inputs=[inputs], outputs=[outputs])
#
#     # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=['accuracy'])
#     # model.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=[dice_coef])
#     # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
#
#     # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mse', dice_coef])
#     # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse', dice_coef])
#     # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mse', dice_coef])
#     model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#
#     model.summary()
#     print('Metrics: ' + str(model.metrics_names))
#     return model
#
# def get_super_small_unet():
#     # batch
#
#     img_shape = (3, IMG_H, IMG_W)
#     concat_axis = 1
#
#     inputs = Input(shape=img_shape)
#     print(inputs.shape)
#
#     conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
#     conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#     print(pool1.shape)
#
#     # conv2 = Conv2D(9, (3, 3), activation='relu', padding='same')(pool1)
#     # conv2 = Conv2D(9, (3, 3), activation='relu', padding='same')(conv2)
#     # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#     # print(pool2.shape)
#
#     conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
#     conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
#     deconv = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='valid')
#     crop = Cropping2D(cropping=((0, 0), (1, 0)))
#     up3 = deconv(conv3)
#     crop_up3 = crop(up3)
#
#     # print(deconv.output_shape, '->', crop.output_shape, conv2.shape)
#     # concat4 = concatenate([crop_up3, conv2], axis=concat_axis)
#     # conv4 = Conv2D(9, (3, 3), activation='relu', padding='same')(concat4)
#     # conv4 = Conv2D(9, (3, 3), activation='relu', padding='same')(conv4)
#     # deconv = Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='valid')
#     # crop = Cropping2D(cropping=((0, 0), (0, 1)))
#     # up4 = deconv(conv4)
#     # crop_up4 = crop(up4)
#
#     print(deconv.output_shape, '->', crop.output_shape, conv1.shape)
#     concat5 = concatenate([crop_up3, conv1], axis=concat_axis)
#     conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(concat5)
#     conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
#     outputs = Conv2D(3, (1, 1), activation='sigmoid')(conv5)
#     print(outputs.shape)
#
#     model = Model(inputs=[inputs], outputs=[outputs])
#
#     # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=['accuracy'])
#     # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
#     # model.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=[dice_coef])
#     # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
#
#     # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mse', dice_coef])
#     # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse', dice_coef])
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mse', dice_coef])
#     # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#
#     model.summary()
#     print('Metrics: ' + str(model.metrics_names))
#     return model
#
# def get_simple_net():
#     # batch
#
#     img_shape = (3, IMG_H, IMG_W)
#     concat_axis = 1
#
#     inputs = Input(shape=img_shape)
#     conv = Conv2D(3, (1, 1), activation='relu')(inputs)
#     outputs = Conv2D(3, (1, 1), activation='sigmoid')(conv)
#
#     # conv1 = Conv2D(3, (3, 3), activation='relu', padding='same')(inputs)
#     # conv1 = Conv2D(3, (3, 3), activation='relu', padding='same')(conv1)
#     # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#     # print(pool1.shape)
#     #
#     # # conv2 = Conv2D(9, (3, 3), activation='relu', padding='same')(pool1)
#     # # conv2 = Conv2D(9, (3, 3), activation='relu', padding='same')(conv2)
#     # # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#     # # print(pool2.shape)
#     #
#     # conv3 = Conv2D(9, (3, 3), activation='relu', padding='same')(pool1)
#     # conv3 = Conv2D(9, (3, 3), activation='relu', padding='same')(conv3)
#     # deconv = Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='valid')
#     # crop = Cropping2D(cropping=((0, 0), (1, 0)))
#     # up3 = deconv(conv3)
#     # crop_up3 = crop(up3)
#     #
#     # # print(deconv.output_shape, '->', crop.output_shape, conv2.shape)
#     # # concat4 = concatenate([crop_up3, conv2], axis=concat_axis)
#     # # conv4 = Conv2D(9, (3, 3), activation='relu', padding='same')(concat4)
#     # # conv4 = Conv2D(9, (3, 3), activation='relu', padding='same')(conv4)
#     # # deconv = Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='valid')
#     # # crop = Cropping2D(cropping=((0, 0), (0, 1)))
#     # # up4 = deconv(conv4)
#     # # crop_up4 = crop(up4)
#     #
#     # print(deconv.output_shape, '->', crop.output_shape, conv1.shape)
#     # concat5 = concatenate([crop_up3, conv1], axis=concat_axis)
#     # conv5 = Conv2D(3, (3, 3), activation='relu', padding='same')(concat5)
#     # conv5 = Conv2D(3, (3, 3), activation='relu', padding='same')(conv5)
#     # outputs = Conv2D(3, (1, 1), activation='sigmoid')(conv5)
#
#
#     model = Model(inputs=[inputs], outputs=[outputs])
#
#     # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=['accuracy'])
#     # model.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=[dice_coef])
#     # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
#
#     # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mse', dice_coef])
#     # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse', dice_coef])
#     # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mse', dice_coef])
#     model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#
#     model.summary()
#     print('Metrics: ' + str(model.metrics_names))
#     return model

def get_unet_128():

    img_shape = (3, IMG_H, IMG_W)
    concat_axis = 1

    inputs = Input(shape=img_shape)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    deconv6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([deconv6, conv4], axis=concat_axis)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    deconv7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([deconv7, conv3], axis=concat_axis)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    deconv8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([deconv8, conv2], axis=concat_axis)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    deconv9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([deconv9, conv1], axis=concat_axis)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    outputs = Conv2D(3, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


def get_unet_128_w_BN_kernel_init():

    img_shape = (3, IMG_H, IMG_W)
    concat_axis = 1

    inputs = Input(shape=img_shape)

    conv1 = Conv2D(32, (3, 3), padding='same', kernel_initializer='glorot_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    conv1 = Activation('relu')(BatchNormalization()(Conv2D(32, (3, 3), padding='same', kernel_initializer='glorot_normal')(conv1)))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Activation('relu')(BatchNormalization()(Conv2D(64, (3, 3), padding='same', kernel_initializer='glorot_normal')(pool1)))
    conv2 = Activation('relu')(BatchNormalization()(Conv2D(64, (3, 3), padding='same', kernel_initializer='glorot_normal')(conv2)))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Activation('relu')(BatchNormalization()(Conv2D(128, (3, 3), padding='same', kernel_initializer='glorot_normal')(pool2)))
    conv3 = Activation('relu')(BatchNormalization()(Conv2D(128, (3, 3), padding='same', kernel_initializer='glorot_normal')(conv3)))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Activation('relu')(BatchNormalization()(Conv2D(256, (3, 3), padding='same', kernel_initializer='glorot_normal')(pool3)))
    conv4 = Activation('relu')(BatchNormalization()(Conv2D(256, (3, 3), padding='same', kernel_initializer='glorot_normal')(conv4)))
    dropout4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(dropout4)

    conv5 = Activation('relu')(BatchNormalization()(Conv2D(512, (3, 3), padding='same', kernel_initializer='glorot_normal')(pool4)))
    conv5 = Activation('relu')(BatchNormalization()(Conv2D(512, (3, 3), padding='same', kernel_initializer='glorot_normal')(conv5)))
    dropout5 = Dropout(0.5)(conv5)

    deconv6 = Activation('relu')(BatchNormalization()(Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_initializer='glorot_normal')(dropout5)))
    # up6 = concatenate([deconv6, conv4], axis=concat_axis)
    up6 = Concatenate(axis=concat_axis)([deconv6, conv4])
    conv6 = Activation('relu')(BatchNormalization()(Conv2D(256, (3, 3), padding='same', kernel_initializer='glorot_normal')(up6)))
    conv6 = Activation('relu')(BatchNormalization()(Conv2D(256, (3, 3), padding='same', kernel_initializer='glorot_normal')(conv6)))

    deconv7 = Activation('relu')(BatchNormalization()(Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer='glorot_normal')(conv6)))
    # up7 = concatenate([deconv7, conv3], axis=concat_axis)
    up7 = Concatenate(axis=concat_axis)([deconv7, conv3])
    conv7 = Activation('relu')(BatchNormalization()(Conv2D(128, (3, 3), padding='same', kernel_initializer='glorot_normal')(up7)))
    conv7 = Activation('relu')(BatchNormalization()(Conv2D(128, (3, 3), padding='same', kernel_initializer='glorot_normal')(conv7)))

    deconv8 = Activation('relu')(BatchNormalization()(Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer='glorot_normal')(conv7)))
    # up8 = concatenate([deconv8, conv2], axis=concat_axis)
    up8 = Concatenate(axis=concat_axis)([deconv8, conv2])
    conv8 = Activation('relu')(BatchNormalization()(Conv2D(64, (3, 3), padding='same', kernel_initializer='glorot_normal')(up8)))
    conv8 = Activation('relu')(BatchNormalization()(Conv2D(64, (3, 3), padding='same', kernel_initializer='glorot_normal')(conv8)))

    deconv9 = Activation('relu')(BatchNormalization()(Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', kernel_initializer='glorot_normal')(conv8)))
    # up9 = concatenate([deconv9, conv1], axis=concat_axis)
    up9 = Concatenate(axis=concat_axis)([deconv9, conv1])
    conv9 = Activation('relu')(BatchNormalization()(Conv2D(32, (3, 3), padding='same', kernel_initializer='glorot_normal')(up9)))
    conv9 = Activation('relu')(BatchNormalization()(Conv2D(32, (3, 3), padding='same', kernel_initializer='glorot_normal')(conv9)))

    outputs = Conv2D(3, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


def get_SPN():

    img_shape = (3, IMG_H, IMG_W)
    concat_axis = 1

    inputs = Input(shape=img_shape)

    # 1-line
    conv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    conv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1_1)
    deconv1_1 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(conv1_1)

    conv2_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(deconv1_1)
    conv2_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv2_1)
    pool2_1 = MaxPooling2D(pool_size=(2, 2))(conv2_1)

    concat3_1 = Concatenate(axis=concat_axis)([conv1_1, pool2_1])
    conv3_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(concat3_1)
    conv3_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3_1)

    # 2-line
    conv1_2 = Conv2D(16, (5, 5), activation='relu', padding='same')(inputs)
    conv1_2 = Conv2D(16, (5, 5), activation='relu', padding='same')(conv1_2)
    deconv1_2 = Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same')(conv1_2)

    conv2_2 = Conv2D(16, (5, 5), activation='relu', padding='same')(deconv1_2)
    conv2_2 = Conv2D(16, (5, 5), activation='relu', padding='same')(conv2_2)
    pool2_2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)

    concat3_2 = Concatenate(axis=concat_axis)([conv1_2, pool2_2])
    conv3_2 = Conv2D(32, (5, 5), activation='relu', padding='same')(concat3_2)
    conv3_2 = Conv2D(32, (5, 5), activation='relu', padding='same')(conv3_2)

    # 3-line
    conv1_3 = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    conv1_3 = Conv2D(32, (1, 1), activation='relu', padding='same')(conv1_3)

    # concat 1,2,3-line in one
    concat_all = Concatenate(axis=concat_axis)([conv3_1, conv3_2, conv1_3])
    conv_all = Conv2D(16, (5, 5), activation='relu', padding='same')(concat_all)
    conv_all = Conv2D(16, (3, 3), activation='relu', padding='same')(conv_all)
    conv_all = Conv2D(16, (1, 1), activation='relu', padding='same')(conv_all)

    outputs = Conv2D(3, (1, 1), activation='sigmoid')(conv_all)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

def get_L15():

    img_shape = (3, IMG_H, IMG_W)
    concat_axis = 1

    inputs = Input(shape=img_shape)

    # 1-line
    conv1 = Conv2D(128, (19, 19), activation='relu', padding='same')(inputs)
    conv2 = Conv2D(320, (1, 1), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(320, (1, 1), activation='relu', padding='same')(conv2)
    conv4 = Conv2D(320, (1, 1), activation='relu', padding='same')(conv3)
    conv5 = Conv2D(128, (1, 1), activation='relu', padding='same')(conv4)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv7 = Conv2D(512, (1, 1), activation='relu', padding='same')(conv6)
    conv8 = Conv2D(128, (5, 5), activation='relu', padding='same')(conv7)
    conv9 = Conv2D(128, (5, 5), activation='relu', padding='same')(conv8)
    conv10 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv9)
    conv11 = Conv2D(128, (5, 5), activation='relu', padding='same')(conv10)
    conv12 = Conv2D(128, (5, 5), activation='relu', padding='same')(conv11)
    conv13 = Conv2D(256, (1, 1), activation='relu', padding='same')(conv12)
    conv14 = Conv2D(64, (7, 7), activation='relu', padding='same')(conv13)
    conv15 = Conv2D(3, (7, 7), activation='sigmoid', padding='same')(conv14)

    model = Model(inputs=[inputs], outputs=[conv15])

    return model


def save_model(model, train_name, iter_num):
    # Save model and weights
    models_dir = work_dir + '/' + train_name + '/saved_models'
    if not os.path.isdir(models_dir):
        os.makedirs(models_dir)

    model_savepath = models_dir + '/iter_' + str(iter_num) + '.h5'
    model.save(model_savepath)
    print(str(datetime.now())+' save model at: {}'.format(model_savepath))

def print_state(process_name, iter, e, epochs, batch_count, N, model, scores):
    res_str = str(datetime.now()) + ' {} iter:{} ep:{}/{} batch_count:{}/{} '.format(process_name, iter, e, epochs, batch_count, N)
    res_str += ' '.join(map(lambda m, t: m + ':' + str(t), model.metrics_names, scores))
    print(res_str)


on_P = False
if on_P:
    work_dir = '/home/cudauser/trained_models'
else:
    work_dir = '/home/doleinik/trained_models'

# IMG_H, IMG_W = (375, 500)
IMG_H, IMG_W = (128, 128)
K.set_image_data_format('channels_first')

if __name__ == '__main__':

    epochs = 1000
    save_model_step = 200

    if on_P:
        factor = 1.5
    else:
        factor = 1

    model_params = {
        'mean_squared_error_lr_0.001': [int(factor * 224), get_unet_128],
        'mean_squared_error_lr_0.00002': [int(factor * 224), get_unet_128],
        'laplacian_gray_loss': [int(factor * 224), get_unet_128],
        'clip_laplacian_color_loss': [int(factor * 224), get_unet_128],
        'sub_loss': [int(factor * 224), get_unet_128],

        'mean_squared_error_lr_0.001_w_BN_kernel_init': [int(factor * 50), get_unet_128_w_BN_kernel_init],

        'spn_mean_squared_error_lr_0.001': [int(factor * 100), get_SPN],
        'spn_cosine_proximity': [int(factor * 100), get_SPN],

        'l15_mean_squared_error_lr_0.001': [int(factor * 40), get_L15]
    }


    train_name = sys.argv[1]
    batch_size = model_params[train_name][0]

    if len(sys.argv)==4: #resume_training
        f_metrics = open(work_dir + '/' + train_name + '/metrics.csv', 'a')  # csv for ploting graph

        iter_num = int(sys.argv[2])
        epoch_start = int(sys.argv[3])

        model_path = work_dir + '/' + train_name + '/saved_models/iter_' + str(iter_num) + '.h5'
        print('Loading model:' + model_path + ' ...')

        custom_objects = {
            'laplacian_gray_loss': laplacian_gray_loss,
            'clip_laplacian_color_loss': clip_laplacian_color_loss,
            'sub_loss': sub_loss
        }
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        # model.compile(optimizer=Adam(2e-6), loss='mean_squared_error', metrics=['accuracy'])

    else:
        train_dir = work_dir + '/' + train_name
        if not os.path.isdir(train_dir):
            os.makedirs(train_dir)
        f_metrics = open(work_dir + '/' + train_name + '/metrics.csv', 'w')  # csv for ploting graph

        iter_num = 0
        epoch_start = 1

        print('Getting model...')
        model = model_params[train_name][1]()

        if train_name == 'mean_squared_error_lr_0.001':
            model.compile(optimizer='adam', loss='mean_squared_error')
        if train_name == 'mean_squared_error_lr_0.00002':
            model.compile(optimizer=Adam(lr=0.00002), loss='mean_squared_error')
        if train_name == 'laplacian_gray_loss': # (делает красным)
            model.compile(optimizer='adam', loss=laplacian_gray_loss)
        if train_name == 'clip_laplacian_color_loss':
            model.compile(optimizer='adam', loss=clip_laplacian_color_loss)
        if train_name == 'sub_loss': #( + - аналогично mse)
            model.compile(optimizer='adam', loss=sub_loss)

        if train_name == 'mean_squared_error_lr_0.001_w_BN_kernel_init':
            model.compile(optimizer='adam', loss='mean_squared_error')

        if train_name == 'spn_mean_squared_error_lr_0.001':
            model.compile(optimizer='adam', loss='mean_squared_error')
        if train_name == 'spn_cosine_proximity':
            model.compile(optimizer='adam', loss='cosine_proximity')

        if train_name == 'l15_mean_squared_error_lr_0.001':
            model.compile(optimizer='adam', loss='mean_squared_error')

    model.summary()
    print('Metrics: ' + str(model.metrics_names))


    N_train = 133527 * 3
    N_val = 5936 * 3
    N_test = 11853 * 3

    if on_P:
        lmdb_path = '/home/cudauser/SharpeningPhoto/lmdb/'
    else:
        lmdb_path = '/home/doleinik/SharpeningPhoto/lmdb/'

    train_paths = [lmdb_path+'train_blur_lmdb_128', lmdb_path+'train_sharp_lmdb_128']
    val_paths = [lmdb_path+'val_blur_lmdb_128', lmdb_path+'val_sharp_lmdb_128']
    test_paths = [lmdb_path + 'test_blur_lmdb_128', lmdb_path + 'test_sharp_lmdb_128']


    print('main params: train_name=' + train_name + ' batch_size=' + str(batch_size) + ' iter_num='+str(iter_num) + ' epoch_start=' + str(epoch_start))
    print('\nRun training...\n')

    for e in range(epoch_start, epochs+1):
        print('Epoch {}/{}'.format(e, epochs))

        train_batch_count = 0
        train_batch_keylists = gen_batch_keylists(N_train, batch_size)

        for train_keylist in train_batch_keylists:
            iter_num += 1
            train_batch_count += len(train_keylist)
            train_blur_data, train_sharp_data = get_data_from_keys(train_paths, train_keylist)
            # print(str(datetime.now())+'    Train...')
            train_scores = model.train_on_batch(train_blur_data, train_sharp_data) # fit, fit_generator, train_on_batch
            if not isinstance(train_scores, (list, tuple)):
                train_scores = [train_scores]
            print_state('training', iter_num, e, epochs, train_batch_count, N_train, model, train_scores)

            # save model
            if((iter_num % save_model_step) == 0):
                save_model(model, train_name, iter_num)

            # score trained model on val data
            val_batch_count = 0
            val_batch_keylists = gen_batch_keylists(N_val, batch_size)
            val_iter_count = 4
            val_scores = []
            for val_iter_id in range(val_iter_count):
                val_batch_count += len(val_batch_keylists[val_iter_id])
                val_blur_data, val_sharp_data = get_data_from_keys(val_paths, val_batch_keylists[val_iter_id])
                # val_score = model.evaluate(val_blur_data, val_sharp_data, batch_size, 0)
                val_score = model.test_on_batch(val_blur_data, val_sharp_data)
                val_scores.append(val_score)
            val_scores = np.array(val_scores)
            val_scores = val_scores.mean(axis=0)
            if not isinstance(val_scores, (list, tuple)):
                val_scores = [val_scores]
            print_state('validation', iter_num, e, epochs, val_batch_count, N_val, model, val_scores)

            # write score to csv
            f_metrics.write(','.join([str(i) for i in [iter_num] + train_scores + val_scores]) + '\n')
            f_metrics.flush()

    # score trained model on test data
    test_batch_count = 0
    test_batch_keylists = gen_batch_keylists(N_test, batch_size)
    test_scores = []
    for test_keylist in test_batch_keylists:
        test_batch_count += len(test_keylist)
        test_blur_data, test_sharp_data = get_data_from_keys(test_paths, test_keylist)
        test_score = model.evaluate(test_blur_data, test_sharp_data, batch_size, 0)
        test_scores.append(test_score)
    test_scores = np.array(test_scores)
    test_scores = test_scores.mean(axis=0)
    print_state('testing', iter_num, '-', '-', test_batch_count, N_test, model, test_scores)
