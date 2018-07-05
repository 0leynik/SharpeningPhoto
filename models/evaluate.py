# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import Conv2D, Input
from keras.optimizers import Adam
import numpy as np
import os
import cv2
import sys
import glob


# saved_weights_path = 'weights_loss_HR/iter_1000.h5'
saved_weights_path = 'weights_loss/iter_850.h5'


def main():
    do_sharp('imgs/_blur.JPG')

    # imgs_dir = get_input_dir_from_argv()
    # do_sharp_dir(imgs_dir)

    print('\nCompleted!')


def predict_model():

    inputs = Input(shape=(None, None, 1))
    conv = Conv2D(128, (9, 9), padding='same',activation='relu')(inputs)
    conv = Conv2D(64, (3, 3), padding='same', activation='relu')(conv)
    outputs = Conv2D(1, (5, 5), padding='same', activation='linear')(conv)

    model = Model(inputs=[inputs], outputs=[outputs])
    # model.compile(optimizer=Adam(lr=0.0003), loss='mean_squared_error')
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def do_sharp(img_path):
    '''
    Increase the original image 2 times the width and height.

    Увеличение исходного изображения в 2 раза по ширине и высоте.

    :param img_path:
    :return:
    '''
    if not os.path.isfile(img_path):
        print('Specify the path to the image! Error: ' + img_path)

    model = predict_model()
    model.load_weights(saved_weights_path)

    img_dir = os.path.dirname(img_path)

    img_name = os.path.splitext(os.path.basename(img_path))
    name = img_name[0]
    ext = img_name[1]

    # bicubic_img_path = os.path.join(img_dir, name + '_bicubic' + ext)
    sharp_img_path = os.path.join(img_dir, name + '_sharp' + ext)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    shape = img.shape

    YCrCb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    Y_img = YCrCb_img[:, :, 0]
    # Y_img = cv2.copyMakeBorder(Y_img, 6, 6, 6, 6, cv2.BORDER_REFLECT_101)

    Y_in = np.zeros((1, Y_img.shape[0], Y_img.shape[1], 1), dtype=np.float32)
    Y_in[0, :, :, 0] = Y_img.astype(np.float32) / 255.

    Y_pred = model.predict(Y_in, batch_size=1) * 255.

    Y_pred = np.clip(Y_pred, 0, 255)
    Y_pred = Y_pred.astype(np.uint8)
    YCrCb_img[:, :, 0] = Y_pred[0, :, :, 0]
    BGR_pred_img = cv2.cvtColor(YCrCb_img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(sharp_img_path, BGR_pred_img)


def get_input_dir_from_argv():
    print(sys.argv)
    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        sys.exit('Error! Directory \"' + input_dir + '\" not found!')
    print('\nImage directory: ' + input_dir + '\n')
    return input_dir


def do_sharp_dir(imgs_dir):
    for img_path in glob.glob(os.path.join(imgs_dir, '*.*')):
        print('Prepare \"' + img_path + '\"...')
        do_sharp(img_path)

if __name__ == "__main__":
    main()
