# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import lmdb
import caffe
import keras

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input,Conv2D,MaxPooling2D,Conv2DTranspose,Cropping2D,concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K


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

    visualize = False
    batch_size = len(keylist)

    blur_data = np.empty((batch_size, 3, 375, 500), dtype=np.uint8)
    sharp_data = np.empty((batch_size, 3, 375, 500), dtype=np.uint8)
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
            ret_data[i][j] = data

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

    K.set_image_data_format('channels_first')

    img_shape = (3, 375, 500)
    concat_axis = 1

    inputs = Input(shape=img_shape)
    print(inputs.shape)

    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print(pool1.shape)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print(pool2.shape)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    print(pool3.shape)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    print(pool4.shape)

    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)
    deconv = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')
    up5 = deconv(conv5)

    print(deconv.output_shape, conv4.shape)
    # crop4 = Cropping2D(cropping=((1,0),(1,0)))(conv4)
    concat6 = concatenate([up5, conv4], axis=concat_axis)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(concat6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
    deconv = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='valid')
    up6 = deconv(conv6)

    print(deconv.output_shape, conv3.shape)
    # crop3 = Cropping2D(cropping=((1,0),(1,0)))(conv3)
    # concat7 = concatenate([up6, crop3], axis=concat_axis)
    concat7 = concatenate([up6, conv3], axis=concat_axis)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(concat7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
    deconv = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='valid')
    crop = Cropping2D(cropping=((0, 0), (1, 0)))
    up7 = deconv(conv7)
    crop_up7 = crop(up7)

    print(deconv.output_shape, '->', crop.output_shape, conv2.shape)
    # crop2 = Cropping2D(cropping=((1, 0), (0, 0)))(conv2)
    # concat8 = concatenate([up7, crop2], axis=concat_axis)
    concat8 = concatenate([crop_up7, conv2], axis=concat_axis)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(concat8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
    deconv = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='valid')
    crop = Cropping2D(cropping=((0, 0), (0, 1)))
    up8 = deconv(conv8)
    crop_up8 = crop(up8)

    print(deconv.output_shape, '->', crop.output_shape, conv1.shape)
    # crop1 = Cropping2D(cropping=((0, 0), (0, 0)))(conv1)
    # concat9 = concatenate([up8, crop1], axis=concat_axis)
    concat9 = concatenate([crop_up8, conv1], axis=concat_axis)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    outputs = Conv2D(3, (1, 1), activation='sigmoid')(conv9)
    print(outputs.shape)

    model = Model(inputs=[inputs], outputs=[outputs])

    # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    # model.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=[dice_coef])
    # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mse', dice_coef])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse', dice_coef])

    model.summary()
    print('Metrics: ' + str(model.metrics_names))
    return model


if __name__ == '__main__':

    lmdb_path = '/home/doleinik/SharpeningPhoto/lmdb/'
    train_paths = [lmdb_path+'train_blur_lmdb', lmdb_path+'train_sharp_lmdb']
    test_paths = [lmdb_path+'test_blur_lmdb', lmdb_path+'test_sharp_lmdb']
    val_paths = [lmdb_path+'val_blur_lmdb', lmdb_path+'val_sharp_lmdb']

    epochs = 100
    batch_size = 8
    N_train = 133527 * 3
    N_test = 11853 * 3
    N_val = 5936 * 3

    print('Getting custom U-Net model...')
    model = get_unet()

    print('\nRun training...\n')

    for e in range(epochs):
        print('Epoch {}/{}'.format(e,epochs))

        train_batch_count = 0
        train_batch_keylists = gen_batch_keylists(N_train, batch_size)

        for train_keylist in train_batch_keylists:

            train_batch_count += len(train_keylist)
            print('Training {:8d}/{}'.format(train_batch_count, N_train))

            # prepare train batch data
            print('Load data...')
            train_blur_data, train_sharp_data = get_data_from_keys(train_paths, train_keylist)

            train_blur_data = train_blur_data.astype('float32')
            train_blur_data /= 255
            train_sharp_data = train_sharp_data.astype('float32')
            train_sharp_data /= 255
            print('Train...')
            # fit, fit_generator, train_on_batch
            train_scores = model.train_on_batch(train_blur_data, train_sharp_data)
            # print result train on batch
            train_s = ''
            for i in range(len(model.metrics)):
                train_s += str(model.metrics[i]) + ':' + str(train_scores[i]) + '  '
            print(train_s)

            # score trained model on val data
            val_batch_count = 0
            val_batch_keylists = gen_batch_keylists(N_val, batch_size)
            val_scores = []
            for val_keylist in val_batch_keylists:
                val_batch_count += len(val_keylist)
                print('Validation {:8d}/{}'.format(val_batch_count, N_val))

                val_blur_data, val_sharp_data = get_data_from_keys(val_paths, val_keylist)

                val_blur_data = val_blur_data.astype('float32')
                val_blur_data /= 255
                val_sharp_data = val_sharp_data.astype('float32')
                val_sharp_data /= 255

                val_score = model.evaluate(val_blur_data, val_sharp_data, verbose=1)
                val_scores.append(val_score)

            val_scores = np.array(val_scores)
            val_scores = val_scores.mean(axis=0)
            val_s = ''
            for i in range(len(model.metrics)):
                val_s += str(model.metrics[i]) + ':' + str(val_scores[i]) + '  '
            print(val_s)

    #
    #
    #
    #
    # # Save model and weights
    # save_dir = os.path.join(os.getcwd(), 'saved_models')
    # if not os.path.isdir(save_dir):
    #     os.makedirs(save_dir)
    #
    # model_name = 'keras_SP_trained_model.h5'
    # model_path = os.path.join(save_dir, model_name)
    # model.save(model_path)
    # print('Saved trained model at %s ' % model_path)
    #
    # # Score trained model
    # count_batch = 0
    # batch_keylists = gen_batch_keylists(N_test, batch_size)
    # for keylist in batch_keylists:
    #     count_batch += len(keylist)
    #     print('{:8d}/{}'.format(count_batch, N_train))
    #
    #     test_blur_data, test_sharp_data = get_data_from_keys(train_paths, keylist)
    #
    #     test_blur_data = test_blur_data.astype('float32')
    #     test_blur_data /= 255
    #     test_sharp_data = test_sharp_data.astype('float32')
    #     test_sharp_data /= 255
    #
    #     scores = model.evaluate(test_blur_data, test_sharp_data, verbose=1)
    #     print('Test loss:', scores[0])
    #     print('Test accuracy:', scores[1])