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


def get_data_from_keys(blur_lmdb_path, sharp_lmdb_path, keylist):
    '''

    :param lmdb_path:
    :param keylist:
    :return: [np_blur_data, np_sharp_data]
    '''

    paths = [blur_lmdb_path, sharp_lmdb_path]
    batch_size = len(keylist)

    blur_data = np.empty((batch_size, 3, 375, 500), dtype=np.uint8)
    sharp_data = np.empty((batch_size, 3, 375, 500), dtype=np.uint8)
    ret_data = [blur_data, sharp_data]


    datum = caffe.proto.caffe_pb2.Datum()
    for i in range(2):

        env = lmdb.open(paths[i], readonly=True)
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


#************ LMDB *************

visualize = False

lmdb_path = '/home/doleinik/SharpeningPhoto/lmdb'
train_path = [lmdb_path+'train_blur_lmdb', lmdb_path+'train_sharp_lmdb']
test_path = [lmdb_path+'test_blur_lmdb', lmdb_path+'test_sharp_lmdb']
val_path = [lmdb_path+'val_blur_lmdb', lmdb_path+'val_sharp_lmdb']





#****************************************



batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = False
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
print(save_dir)
model_name = 'keras_cifar10_trained_model.h5'

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
# x_train(n_imgs, H, W, C) RBG-images
print('y_train shape:', y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

for i in range(20):
    plt.figure(i)
    plt.imshow(x_train[i, ...])

plt.show()

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
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

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print(x_train.dtype)

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=int(np.ceil(x_train.shape[0] / float(batch_size))),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])