import keras

from keras.layers import Dense,Input,Conv2D,MaxPooling2D,Conv2DTranspose,Cropping2D,concatenate
from keras.models import Model,Sequential
from keras.optimizers import Adam
from keras import backend as K

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
    print(inputs.shape)

    conv1 = Conv2D(3, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(3, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print(pool1.shape)

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

    print(deconv.output_shape, '->', crop.output_shape, conv1.shape)
    concat5 = concatenate([crop_up3, conv1], axis=concat_axis)
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


if __name__ == '__main__':
    # get_unet()
    # binary()
    # get_small_unet()
    get_super_small_unet()