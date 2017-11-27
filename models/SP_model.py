# -*- coding: utf-8 -*-

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import os

import matplotlib.pyplot as plt
import cv2
import lmdb
import caffe


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

            print(type(data))
            print(data.dtype)
            print(data.shape)
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


lmdb_path = '/home/doleinik/SharpeningPhoto/lmdb'
train_paths = [lmdb_path+'train_blur_lmdb', lmdb_path+'train_sharp_lmdb']
test_paths = [lmdb_path+'test_blur_lmdb', lmdb_path+'test_sharp_lmdb']
val_paths = [lmdb_path+'val_blur_lmdb', lmdb_path+'val_sharp_lmdb']

epochs = 100
batch_size = 1024
N_train = 133527 * 3
N_test = 11853 * 3
N_val = 5936 * 3


# NET
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(3,375,500)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


model.compile(keras.optimizers.SGD,
              keras.losses.MSE,
              ['accuracy'])


for e in range(epochs):
    print('Epoch {}/{}'.format(e,epochs))

    train_batch_count = 0
    train_batch_keylists = gen_batch_keylists(N_train, batch_size)

    for train_keylist in train_batch_keylists:

        train_batch_count += len(train_keylist)
        print('Training {:8d}/{}'.format(train_batch_count, N_train))

        # prepare train batch data
        train_blur_data, train_sharp_data = get_data_from_keys(train_paths, train_keylist)

        train_blur_data = train_blur_data.astype('float32')
        train_blur_data /= 255
        train_sharp_data = train_sharp_data.astype('float32')
        train_sharp_data /= 255

        # fit, fit_generator, train_on_batch
        model.train_on_batch(train_blur_data, train_sharp_data)


        # score trained model on val data
        val_batch_count = 0
        val_batch_keylists = gen_batch_keylists(N_val, batch_size)
        for val_keylist in range(len(val_batch_keylists)):
            val_batch_count += len(val_keylist)
            print('Validation {:8d}/{}'.format(val_batch_count, N_val))

            val_blur_data, val_sharp_data = get_data_from_keys(val_paths, val_keylist)

            val_blur_data = val_blur_data.astype('float32')
            val_blur_data /= 255
            val_sharp_data = val_sharp_data.astype('float32')
            val_sharp_data /= 255

            scores = model.evaluate(val_blur_data, val_sharp_data, verbose=1)
            print('Val loss:', scores[0])
            print('Val accuracy:', scores[1])




# Save model and weights
save_dir = os.path.join(os.getcwd(), 'saved_models')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

model_name = 'keras_SP_trained_model.h5'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model
count_batch = 0
batch_keylists = gen_batch_keylists(N_test, batch_size)
for keylist in batch_keylists:
    count_batch += len(keylist)
    print('{:8d}/{}'.format(count_batch, N_train))

    test_blur_data, test_sharp_data = get_data_from_keys(train_paths, keylist)

    test_blur_data = test_blur_data.astype('float32')
    test_blur_data /= 255
    test_sharp_data = test_sharp_data.astype('float32')
    test_sharp_data /= 255

    scores = model.evaluate(test_blur_data, test_sharp_data, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])