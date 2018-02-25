# -*- coding: utf-8 -*-

from __future__ import print_function

import keras
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import lmdb
import caffe
from datetime import datetime

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input,Conv2D,MaxPooling2D,Conv2DTranspose,Cropping2D,concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras import backend as K

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

limit_mem = False
if limit_mem:
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.Session(config=config))

# IMG_H, IMG_W = (375, 500)
IMG_H, IMG_W = (128, 128)
K.set_image_data_format('channels_first')

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


# https://www.kaggle.com/toregil/a-lung-u-net-in-keras/notebook
# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def get_unet():
    # batch 170

    img_shape = (3, IMG_H, IMG_W)
    concat_axis = 1

    inputs = Input(shape=img_shape)
    print(inputs.shape)

    conv1 = Conv2D(2, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(2, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print(pool1.shape)

    conv2 = Conv2D(4, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(4, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print(pool2.shape)

    conv3 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    print(pool3.shape)

    conv4 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    print(pool4.shape)

    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    deconv = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')
    up5 = deconv(conv5)

    print(deconv.output_shape, conv4.shape)
    # crop4 = Cropping2D(cropping=((1,0),(1,0)))(conv4)
    concat6 = concatenate([up5, conv4], axis=concat_axis)
    conv6 = Conv2D(16, (3, 3), activation='relu', padding='same')(concat6)
    conv6 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv6)
    deconv = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='valid')
    up6 = deconv(conv6)

    print(deconv.output_shape, conv3.shape)
    # crop3 = Cropping2D(cropping=((1,0),(1,0)))(conv3)
    # concat7 = concatenate([up6, crop3], axis=concat_axis)
    concat7 = concatenate([up6, conv3], axis=concat_axis)
    conv7 = Conv2D(8, (3, 3), activation='relu', padding='same')(concat7)
    conv7 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv7)
    deconv = Conv2DTranspose(4, (3, 3), strides=(2, 2), padding='valid')
    crop = Cropping2D(cropping=((0, 0), (1, 0)))
    up7 = deconv(conv7)
    crop_up7 = crop(up7)

    print(deconv.output_shape, '->', crop.output_shape, conv2.shape)
    # crop2 = Cropping2D(cropping=((1, 0), (0, 0)))(conv2)
    # concat8 = concatenate([up7, crop2], axis=concat_axis)
    concat8 = concatenate([crop_up7, conv2], axis=concat_axis)
    conv8 = Conv2D(4, (3, 3), activation='relu', padding='same')(concat8)
    conv8 = Conv2D(4, (3, 3), activation='relu', padding='same')(conv8)
    deconv = Conv2DTranspose(2, (3, 3), strides=(2, 2), padding='valid')
    crop = Cropping2D(cropping=((0, 0), (0, 1)))
    up8 = deconv(conv8)
    crop_up8 = crop(up8)

    print(deconv.output_shape, '->', crop.output_shape, conv1.shape)
    # crop1 = Cropping2D(cropping=((0, 0), (0, 0)))(conv1)
    # concat9 = concatenate([up8, crop1], axis=concat_axis)
    concat9 = concatenate([crop_up8, conv1], axis=concat_axis)
    conv9 = Conv2D(2, (3, 3), activation='relu', padding='same')(concat9)
    conv9 = Conv2D(2, (3, 3), activation='relu', padding='same')(conv9)
    outputs = Conv2D(3, (1, 1), activation='sigmoid')(conv9)
    print(outputs.shape)

    model = Model(inputs=[inputs], outputs=[outputs])

    # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=['accuracy'])
    # model.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=[dice_coef])
    # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mse', dice_coef])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse', dice_coef])
    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mse', dice_coef])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    model.summary()
    print('Metrics: ' + str(model.metrics_names))
    return model

def get_small_unet():
    # batch 176

    img_shape = (3, IMG_H, IMG_W)
    concat_axis = 1

    inputs = Input(shape=img_shape)
    print(inputs.shape)

    conv1 = Conv2D(3, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(3, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print(pool1.shape)

    conv2 = Conv2D(9, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(9, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print(pool2.shape)

    conv3 = Conv2D(18, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(18, (3, 3), activation='relu', padding='same')(conv3)
    deconv = Conv2DTranspose(9, (3, 3), strides=(2, 2), padding='valid')
    crop = Cropping2D(cropping=((0, 0), (1, 0)))
    up3 = deconv(conv3)
    crop_up3 = crop(up3)

    print(deconv.output_shape, '->', crop.output_shape, conv2.shape)
    concat4 = concatenate([crop_up3, conv2], axis=concat_axis)
    conv4 = Conv2D(9, (3, 3), activation='relu', padding='same')(concat4)
    conv4 = Conv2D(9, (3, 3), activation='relu', padding='same')(conv4)
    deconv = Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='valid')
    crop = Cropping2D(cropping=((0, 0), (0, 1)))
    up4 = deconv(conv4)
    crop_up4 = crop(up4)

    print(deconv.output_shape, '->', crop.output_shape, conv1.shape)
    concat5 = concatenate([crop_up4, conv1], axis=concat_axis)
    conv5 = Conv2D(3, (3, 3), activation='relu', padding='same')(concat5)
    conv5 = Conv2D(3, (3, 3), activation='relu', padding='same')(conv5)
    outputs = Conv2D(3, (1, 1), activation='sigmoid')(conv5)
    print(outputs.shape)

    model = Model(inputs=[inputs], outputs=[outputs])

    # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=['accuracy'])
    # model.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=[dice_coef])
    # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mse', dice_coef])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse', dice_coef])
    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mse', dice_coef])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    model.summary()
    print('Metrics: ' + str(model.metrics_names))
    return model

def get_super_small_unet():
    # batch

    img_shape = (3, IMG_H, IMG_W)
    concat_axis = 1

    inputs = Input(shape=img_shape)
    print(inputs.shape)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print(pool1.shape)

    # conv2 = Conv2D(9, (3, 3), activation='relu', padding='same')(pool1)
    # conv2 = Conv2D(9, (3, 3), activation='relu', padding='same')(conv2)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # print(pool2.shape)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    deconv = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='valid')
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

    print(deconv.output_shape, '->', crop.output_shape, conv1.shape)
    concat5 = concatenate([crop_up3, conv1], axis=concat_axis)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(concat5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    outputs = Conv2D(3, (1, 1), activation='sigmoid')(conv5)
    print(outputs.shape)

    model = Model(inputs=[inputs], outputs=[outputs])

    # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    # model.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=[dice_coef])
    # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mse', dice_coef])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse', dice_coef])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mse', dice_coef])
    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    model.summary()
    print('Metrics: ' + str(model.metrics_names))
    return model

def get_simple_net():
    # batch

    img_shape = (3, IMG_H, IMG_W)
    concat_axis = 1

    inputs = Input(shape=img_shape)
    conv = Conv2D(3, (1, 1), activation='relu')(inputs)
    outputs = Conv2D(3, (1, 1), activation='sigmoid')(conv)

    # conv1 = Conv2D(3, (3, 3), activation='relu', padding='same')(inputs)
    # conv1 = Conv2D(3, (3, 3), activation='relu', padding='same')(conv1)
    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # print(pool1.shape)
    #
    # # conv2 = Conv2D(9, (3, 3), activation='relu', padding='same')(pool1)
    # # conv2 = Conv2D(9, (3, 3), activation='relu', padding='same')(conv2)
    # # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # # print(pool2.shape)
    #
    # conv3 = Conv2D(9, (3, 3), activation='relu', padding='same')(pool1)
    # conv3 = Conv2D(9, (3, 3), activation='relu', padding='same')(conv3)
    # deconv = Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='valid')
    # crop = Cropping2D(cropping=((0, 0), (1, 0)))
    # up3 = deconv(conv3)
    # crop_up3 = crop(up3)
    #
    # # print(deconv.output_shape, '->', crop.output_shape, conv2.shape)
    # # concat4 = concatenate([crop_up3, conv2], axis=concat_axis)
    # # conv4 = Conv2D(9, (3, 3), activation='relu', padding='same')(concat4)
    # # conv4 = Conv2D(9, (3, 3), activation='relu', padding='same')(conv4)
    # # deconv = Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='valid')
    # # crop = Cropping2D(cropping=((0, 0), (0, 1)))
    # # up4 = deconv(conv4)
    # # crop_up4 = crop(up4)
    #
    # print(deconv.output_shape, '->', crop.output_shape, conv1.shape)
    # concat5 = concatenate([crop_up3, conv1], axis=concat_axis)
    # conv5 = Conv2D(3, (3, 3), activation='relu', padding='same')(concat5)
    # conv5 = Conv2D(3, (3, 3), activation='relu', padding='same')(conv5)
    # outputs = Conv2D(3, (1, 1), activation='sigmoid')(conv5)


    model = Model(inputs=[inputs], outputs=[outputs])

    # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=['accuracy'])
    # model.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=[dice_coef])
    # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mse', dice_coef])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse', dice_coef])
    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mse', dice_coef])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    model.summary()
    print('Metrics: ' + str(model.metrics_names))
    return model

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

    # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=['accuracy'])
    # model.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=[dice_coef])
    # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mse', dice_coef])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse', dice_coef])
    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mse', dice_coef])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    model.summary()
    print('Metrics: ' + str(model.metrics_names))

    return model


def save_model(model, iter_num):
    # Save model and weights
    save_dir = '/home/doleinik/SP_saved_models'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model_name = 'SP_model_iter_' + str(iter_num) + '.h5'
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print(str(datetime.now())+' save model at:{}'.format(model_path))

def print_state(process_name, iter, e, epochs, batch_count, N, model, scores):
    res_str = str(datetime.now()) + ' {} iter:{} ep:{}/{} batch_count:{}/{}'.format(process_name, iter, e, epochs, batch_count, N)
    res_str += ' '.join(map(lambda m, t: m + ':' + str(t), model.metrics_names, scores))
    print(res_str)



if __name__ == '__main__':

    lmdb_path = '/home/doleinik/SharpeningPhoto/lmdb/'
    train_paths = [lmdb_path+'train_blur_lmdb_128', lmdb_path+'train_sharp_lmdb_128']
    test_paths = [lmdb_path+'test_blur_lmdb_128', lmdb_path+'test_sharp_lmdb_128']
    val_paths = [lmdb_path+'val_blur_lmdb_128', lmdb_path+'val_sharp_lmdb_128']

    epochs = 1000
    batch_size = 224
    save_model_step = 500

    N_train = 133527 * 3
    N_test = 11853 * 3
    N_val = 5936 * 3

    # resume training
    if True:
        iter_num = 39500
        model_path = '/home/doleinik/SP_saved_models/SP_model_iter_39500.h5'
        print('Loading model:' + model_path + ' ...')
        model = keras.models.load_model(model_path)
        f_metrics = open('/home/doleinik/SP_metrics.csv', 'a') # csv for ploting graph
    else:
        iter_num = 0
        print('Getting model...')
        model = get_unet_128()
        f_metrics = open('SP_metrics.csv', 'w') # csv for ploting graph

    # print('\nRun training...\n')
    #
    # for e in range(1, epochs+1):
    #     print('Epoch {}/{}'.format(e, epochs))
    #
    #     train_batch_count = 0
    #     train_batch_keylists = gen_batch_keylists(N_train, batch_size)
    #
    #     for train_keylist in train_batch_keylists:
    #         iter_num += 1
    #         train_batch_count += len(train_keylist)
    #
    #         train_blur_data, train_sharp_data = get_data_from_keys(train_paths, train_keylist)
    #
    #         # print(str(datetime.now())+'    Train...')
    #         train_scores = model.train_on_batch(train_blur_data, train_sharp_data) # fit, fit_generator, train_on_batch
    #
    #         print_state('training', iter_num, e, epochs, train_batch_count, N_train, model, train_scores)
    #
    #         # write score to csv
    #         f_metrics.write(','.join([str(i) for i in [iter_num]+train_scores]) + '\n')
    #
    #         # save model
    #         if((iter_num % save_model_step) == 0):
    #             save_model(model, iter_num)
    #

    # score trained model on val data
    val_batch_count = 0
    val_batch_keylists = gen_batch_keylists(N_val, batch_size)
    val_scores = []
    for val_keylist in val_batch_keylists:
        val_batch_count += len(val_keylist)
        val_blur_data, val_sharp_data = get_data_from_keys(val_paths, val_keylist)
        val_score = model.evaluate(val_blur_data, val_sharp_data, verbose=1)
        val_scores.append(val_score)
    val_scores = np.array(val_scores)
    val_scores = val_scores.mean(axis=0)
    # print_state('validation', iter_num, e, epochs, val_batch_count, N_val, model, val_scores)
    print_state('validation', iter_num, '-', '-', val_batch_count, N_val, model, val_scores)

    # score trained model on test data
    test_batch_count = 0
    test_batch_keylists = gen_batch_keylists(N_test, batch_size)
    test_scores = []
    for test_keylist in test_batch_keylists:
        test_batch_count += len(test_keylist)
        test_blur_data, test_sharp_data = get_data_from_keys(test_paths, test_keylist)
        test_score = model.evaluate(test_blur_data, test_sharp_data, verbose=1)
        test_scores.append(test_score)
    test_scores = np.array(test_scores)
    test_scores = test_scores.mean(axis=0)
    print_state('testing', iter_num, '-', '-', test_batch_count, N_test, model, test_scores)