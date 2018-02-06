# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import lmdb
import caffe
from datetime import datetime
import keras

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input,Conv2D,MaxPooling2D,Conv2DTranspose,Cropping2D,concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras import backend as K

import SP_model

if __name__ == '__main__':




    model = SP_model.get_unet_128()
    model.load_weights('lung.h5')

    plt.plot(hist.history['loss'], color='b')
    plt.plot(hist.history['val_loss'], color='r')
    plt.show()
    plt.plot(hist.history['dice_coef'], color='b')
    plt.plot(hist.history['val_dice_coef'], color='r')
    plt.show()

    y_hat = model.predict(x_val)
    fig, ax = plt.subplots(1,3,figsize=(12,6))
    ax[0].imshow(x_val[0,:,:,0], cmap='gray')
    ax[1].imshow(y_val[0,:,:,0])
    ax[2].imshow(y_hat[0,:,:,0])