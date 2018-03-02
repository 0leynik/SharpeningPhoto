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

K.set_image_data_format('channels_first')


BGR = [0.114, 0.587, 0.299]

def test_metric():
    img = skimage.img_as_float(imread('/Users/dmitryoleynik/PycharmProjects/SharpeningPhoto/filter_ImageNet_for_visio/4.JPEG'))[:150, :128] # RGB
    h, w, c = img.shape
    img = img[..., ::-1]  # BGR
    img = np.transpose(img, (2, 0, 1))
    # print(img.shape)


    img_arr = np.empty((10, 3, h, w))
    img_arr[0] = img

    y_true = K.variable(img_arr)
    y_pred = K.variable(img_arr)


    y_true_gray = y_true[:, 0:1] * BGR[0] + y_true[:, 1:2] * BGR[1] + y_true[:, 2:3] * BGR[2] # to GRAY
    y_pred_gray = y_pred[:, 0:1] * BGR[0] + y_pred[:, 1:2] * BGR[1] + y_pred[:, 2:3] * BGR[2]  # to GRAY
    print(y_true_gray.shape)

    kernel = K.variable(np.array([[[[-1]], [[-1]], [[-1]]], [[[-1]], [[8]], [[-1]]], [[[-1]], [[-1]], [[-1]]]]), dtype='float32')

    y_true_conv = K.conv2d(y_true_gray, kernel, (1, 1), 'same', 'channels_first') # edge detection with Laplacian
    y_true_conv = K.clip(y_true_conv, 0, 1)

    y_pred_conv = K.conv2d(y_pred_gray, kernel, (1, 1), 'same', 'channels_first') # edge detection with Laplacian
    y_pred_conv = K.clip(y_pred_conv, 0, 1)
    print(y_pred_conv.shape)


    abs = K.abs(y_pred_conv - y_true_conv)
    print(abs.shape)

    mean = K.mean(abs)
    print('mean: ' + str(mean.shape))



    img_lapl = K.eval(y_true_conv)[0][0]
    # print(img_lapl.shape)
    img_lapl = np.clip(img_lapl, 0, 1)
    # plt.imshow(img_lapl, cmap='gray')
    # plt.show()

    cv2.imshow('image', img_lapl)
    cv2.waitKey()


def colour_metric():
    pass




def dice_coef(y_true, y_pred):

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # median = cv2.medianBlur(gray, ksize_median)
    # calc_laplacian = cv2.Laplacian(median, -1, ksize=ksize_laplacian)
    # inverts = cv2.bitwise_not(calc_laplacian)
    # std_blured_images.append(calc_laplacian.std())
    # values.append(laplacian.mean())

    # print(np.mean(y_truee))

    # intersection = K.sum(y_true_f * y_pred_f)
    # kek = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    # print(kek.shape)
    # return K.mean(y_true)
    # y_pred = np.clip(y_pred, K.epsilon(), 1.0 - K.mapepsilon())
    # out = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    # return np.mean(out, axis=-1)

    gray_true = K.stack([y_true[:, 0] * BGR[0], y_true[:, 1] * BGR[1], y_true[:, 2] * BGR[2]], axis=1)
    gray_true = K.stack([y_true[:, 0] * BGR[0], y_true[:, 1] * BGR[1], y_true[:, 2] * BGR[2]], axis=1)


    kernel = K.variable(np.array([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]*3]*3), dtype='float32')
    y_true = K.conv2d(gray_true, kernel, (1, 1), 'same', 'channels_first')
    # print (y_true.shape)
    # print (K.mean(y_true).shape)
    return K.mean(y_true)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


BGR = [0.114, 0.587, 0.299]
def laplacian_loss(y_true, y_pred):
    y_true_gray = y_true[:, 0:1] * BGR[0] + y_true[:, 1:2] * BGR[1] + y_true[:, 2:3] * BGR[2]  # to GRAY
    y_pred_gray = y_pred[:, 0:1] * BGR[0] + y_pred[:, 1:2] * BGR[1] + y_pred[:, 2:3] * BGR[2]  # to GRAY
    print(y_true_gray.shape)

    kernel = K.variable(np.array([[[[-1]], [[-1]], [[-1]]], [[[-1]], [[8]], [[-1]]], [[[-1]], [[-1]], [[-1]]]]),
                        dtype='float32')

    y_true_conv = K.conv2d(y_true_gray, kernel, (1, 1), 'same', 'channels_first')  # edge detection with Laplacian
    y_true_conv = K.clip(y_true_conv, 0, 1)

    y_pred_conv = K.conv2d(y_pred_gray, kernel, (1, 1), 'same', 'channels_first')  # edge detection with Laplacian
    y_pred_conv = K.clip(y_pred_conv, 0, 1)
    print(y_pred_conv.shape)

    abs = K.abs(y_pred_conv - y_true_conv)
    print(abs.shape)

    mean = K.mean(abs)
    print(mean.shape)

    return mean

def get_unet():

    K.set_image_data_format('channels_first')

    img_shape = (3, 375, 500)
    concat_axis = 1

    inputs = Input(shape=img_shape)
    print inputs.shape

    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print pool1.shape

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print pool2.shape

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    print pool3.shape

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    print pool4.shape

    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)
    deconv = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')
    up5 = deconv(conv5)

    print deconv.output_shape, conv4.shape
    # crop4 = Cropping2D(cropping=((1,0),(1,0)))(conv4)
    concat6 = concatenate([up5, conv4], axis=concat_axis)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(concat6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
    deconv = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='valid')
    up6 = deconv(conv6)

    print deconv.output_shape, conv3.shape
    # crop3 = Cropping2D(cropping=((1,0),(1,0)))(conv3)
    # concat7 = concatenate([up6, crop3], axis=concat_axis)
    concat7 = concatenate([up6, conv3], axis=concat_axis)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(concat7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
    deconv = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='valid')
    crop = Cropping2D(cropping=((0, 0), (1, 0)))
    up7 = deconv(conv7)
    crop_up7 = crop(up7)

    print deconv.output_shape, '->', crop.output_shape, conv2.shape
    # crop2 = Cropping2D(cropping=((1, 0), (0, 0)))(conv2)
    # concat8 = concatenate([up7, crop2], axis=concat_axis)
    concat8 = concatenate([crop_up7, conv2], axis=concat_axis)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(concat8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
    deconv = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='valid')
    crop = Cropping2D(cropping=((0, 0), (0, 1)))
    up8 = deconv(conv8)
    crop_up8 = crop(up8)

    print deconv.output_shape, '->', crop.output_shape, conv1.shape
    # crop1 = Cropping2D(cropping=((0, 0), (0, 0)))(conv1)
    # concat9 = concatenate([up8, crop1], axis=concat_axis)
    concat9 = concatenate([crop_up8, conv1], axis=concat_axis)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    outputs = Conv2D(3, (1, 1), activation='sigmoid')(conv9)
    print outputs.shape

    model = Model(inputs=[inputs], outputs=[outputs])

    # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    # model.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=[dice_coef])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mse', dice_coef])

    model.summary()

    return model

def binary():
    # For a single-input model with 2 classes (binary classification):

    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Generate dummy data
    import numpy as np
    data = np.random.random((1000, 100))
    labels = np.random.randint(2, size=(1000, 1))

    # Train the model, iterating on the data in batches of 32 samples
    model.fit(data, labels, epochs=100, batch_size=32)

def get_small_unet():
    # batch 170
    K.set_image_data_format('channels_first')

    img_shape = (3, 375, 500)
    concat_axis = 1

    # deconv = Conv2DTranspose(9, (2, 2), strides=(2, 2), padding='same')
    # # deconv = Conv2DTranspose(4, (3, 3), strides=(2, 2), padding='valid')
    # crop = Cropping2D(cropping=((0, 0), (1, 0)))
    # up7 = deconv(conv7)
    # crop_up7 = crop(up7)

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

    conv3 = Conv2D(27, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(27, (3, 3), activation='relu', padding='same')(conv3)
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

    # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=['accuracy'])
    # model.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=[dice_coef])
    # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mse', dice_coef])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse', dice_coef])
    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mse', dice_coef])
    model.compile(optimizer='adam', loss=laplacian_loss, metrics=['accuracy', dice_coef])

    model.summary()
    # print('Metrics: ' + str(model.metrics_names))
    return model


if __name__ == '__main__':
    # get_unet()
    # binary()
    # get_small_unet()

    # test_metric()

    model = get_super_small_unet()

    arr = np.empty((10,3,375,500), dtype=np.float32)
    scores = model.train_on_batch(arr, arr)

    for m in zip(model.metrics_names,scores):
        print(m)
