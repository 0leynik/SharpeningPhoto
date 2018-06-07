# -*- coding: utf-8 -*-

from __future__ import print_function

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import cv2

from keras.layers import Input,Conv2D,MaxPooling2D,Conv2DTranspose, concatenate
from keras.models import Model
import keras.backend as K
K.set_image_data_format('channels_first')


def get_deploy_unet():

    img_shape = (3, None, None)
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

    deconv6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), activation='relu', padding='same')(conv5)
    up6 = concatenate([deconv6, conv4], axis=concat_axis)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    deconv7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), activation='relu', padding='same')(conv6)
    up7 = concatenate([deconv7, conv3], axis=concat_axis)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    deconv8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), activation='relu', padding='same')(conv7)
    up8 = concatenate([deconv8, conv2], axis=concat_axis)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    deconv9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), activation='relu', padding='same')(conv8)
    up9 = concatenate([deconv9, conv1], axis=concat_axis)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    outputs = Conv2D(3, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def predict(input_dir, output_dir):
    model = get_deploy_unet()
    # model.summary()
    model.load_weights('model.h5')

    for img_name in os.listdir(input_dir):
        print('Обработка файла \"' + img_name + '\"...')
        split = os.path.splitext(img_name)
        name = split[0]
        ext = split[1]


        img = cv2.imread(os.path.join(input_dir,img_name))/255.  #BGR
        h, w, c = img.shape
        # print('shape img: ' + str(img.shape))

        img = np.transpose(img, (2, 0, 1))  # HxWxC -> CxHxW

        img_to_pred = np.empty((1, c, 128 * (h/128 + 1), 128 * (w/128 + 1)))
        img_to_pred[0, :, :h, :w] = img
        # print('shape pred: ' + str(img_to_pred.shape))

        predict_img = model.predict(img_to_pred)

        predict_img = np.transpose(predict_img[0, :, :h, :w], (1, 2, 0))  # CxHxW -> HxWxC
        predict_img = predict_img * 255
        predict_img = np.clip(predict_img, 0, 255)
        predict_img = predict_img.astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, name+'_sharp'+ext if input_dir == output_dir else img_name), predict_img)


if __name__ == '__main__':

    if os.environ.has_key('SP_INPUT_IMG_DIR'):
        input_dir = os.path.abspath(os.environ['SP_INPUT_IMG_DIR'])
        if not os.path.isdir(input_dir):
            print('Ошибка! Директория \"' + input_dir + '\" не надена!')
            exit()
    else:
        print('Ошибка! SP_INPUT_IMG_DIR не найден!')

    if os.environ.has_key('SP_OUTPUT_IMG_DIR'):
        output_dir = os.path.abspath(os.environ['SP_OUTPUT_IMG_DIR'])
        if output_dir=='':
            print('Ошибка! Неправильно указана директория для сохранения!')
            exit()
        else:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

    else:
        output_dir = input_dir

    print('Директория входных изображений: ' + input_dir)
    print('Директория обработанных изображений: ' + output_dir)

    predict(input_dir, output_dir)

    print('\nЗавершено!')
