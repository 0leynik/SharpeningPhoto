# -*- coding: utf-8 -*-

from __future__ import print_function

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.io import imread, imshow, imsave
import skimage
import keras
import keras.backend as K
import SP_model
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d


def graph_metrics(train_name):
    model_path = '/home/doleinik/trained_models_SharpeningPhoto/' + train_name + '/SP_metrics.csv'

    # metrics = np.loadtxt(os.path.expanduser('~/SP_metrics.csv'), delimiter=',')
    metrics = np.loadtxt(model_path, delimiter=',')

    save_dir_graphs = '/home/doleinik/trained_models_SharpeningPhoto/graphs/'
    if not os.path.isdir(save_dir_graphs):
        os.makedirs(save_dir_graphs)

    loss_name = 'loss_' + train_name
    plt.figure(loss_name)
    plt.title(loss_name)
    plt.plot(metrics[:, 1])
    plt.ylabel('loss')
    plt.xlabel('iter')
    plt.grid(True, linestyle='--')
    # plt.yticks(np.linspace(0., 0.2, 10))
    plt.ylim(0., 0.25)
    plt.savefig(save_dir_graphs + loss_name + '.png')
    plt.close()

    acc_name = 'acc_' + train_name
    plt.figure(acc_name)
    plt.title(acc_name)
    plt.plot(metrics[:, 2])
    plt.ylabel('acc')
    plt.xlabel('iter')
    plt.grid(True, linestyle='--')
    # plt.yticks(np.linspace(0., 0.2, 10))
    plt.ylim(0., 0.2)
    plt.savefig(save_dir_graphs + acc_name + '.png')
    plt.close()

    # plt.show()


def plt_img(data):
    # CxHxW -> HxWxC
    img = np.transpose(data, (1, 2, 0))

    # BGR -> RGB
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[:, :, ::-1]

    # matplotlib.pyplot.imshow()
    # HxWx3 – RGB (float or uint8 array)
    plt.imshow(img)
    return img


h_step, w_step = (128, 128)
h_count = 0
w_count = 0

def get_patches_from_img(img):
    global h_count, w_count
    h, w, c = img.shape

    h_count = h / h_step
    if (h % h_step) != 0:
        h_count = h / h_step + 1
    h_new = h_count * h_step

    w_count = w / w_step
    if (w % w_step) != 0:
        w_count = w / w_step + 1
    w_new = w_count * w_step

    img_new = np.zeros((h_new, w_new, c), dtype=np.float32)
    img_new[:h, :w] = img

    i = 0
    patches = np.zeros((h_count * w_count, h_step, w_step, c), dtype=np.float32)

    for height in np.split(img_new, h_count, axis=0):
        print(height.shape)
        for width in np.split(height, w_count, axis=1):
            print(width.shape)
            patches[i] = width
            i += 1

    return patches

def get_img_from_patches(patches, img):
    h, w, c = img.shape
    b_count = patches.shape[0]
    print(patches.shape)
    extended_img = np.zeros((h_count * h_step, w_count * w_step, c), dtype=np.float32)
    k = 0
    for i in range(h_count):
        for j in range(w_count):
            extended_img[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step] = patches[k]
            k += 1

    return extended_img[:h, :w]


def evaluate(train_name, iter_num):

    # cluster run
    model_path = '/home/doleinik/trained_models_SharpeningPhoto/' + train_name + '/SP_saved_models/SP_model_iter_' + str(iter_num) + '.h5'

    # loacal
    # model_path = '/Users/dmitryoleynik/PycharmProjects/SharpeningPhoto/models/SP_model_iter_39500_mse.h5'
    # model_path = '/Users/dmitryoleynik/PycharmProjects/SharpeningPhoto/models/SP_model_iter_39500_sub_loss.h5'

    # load model
    custom_objects = {
        'laplacian_gray_loss': SP_model.laplacian_gray_loss,
        'sub_loss' : SP_model.sub_loss,
        'clip_laplacian_color_loss' : SP_model.clip_laplacian_color_loss
    }
    model = keras.models.load_model(model_path, custom_objects=custom_objects)


    # execute for single image file
    if False:
        # img_path = '/home/doleinik/SharpeningPhoto/quality_ImageNet/test_500/images/100_fb.JPEG'
        # img_path = '/home/doleinik/me.jpg'
        img_path = '9_1.JPG'

        original_img = skimage.img_as_float(imread(img_path))
        # original_img = skimage.img_as_float(imread(img_path))[:128,:128]
        print(original_img.shape)
        img = original_img[:, :, ::-1]  # RGB -> BGR
        img_patches = get_patches_from_img(img)
        img_patches = np.array(map(lambda i: np.transpose(i, (2, 0, 1)), img_patches), np.float32)  # HxWxC -> CxHxW
        print('shape img_patches: ' + str(img_patches.shape))

        # np_img = np.empty((1, 3, 128, 128), dtype=np.float32)
        # np_img[0] = np.transpose(img, (2, 0, 1))
        print('Predicting...')
        # predict_img = model.predict(np_img)
        predict_img_patches = model.predict(img_patches)

        predict_img_patches = np.array(map(lambda i: np.transpose(i, (1, 2, 0)), predict_img_patches), np.float32)  # HxWxC -> CxHxW
        predict_img = get_img_from_patches(predict_img_patches, original_img)
        predict_img = predict_img[:, :, ::-1]  # BGR -> RGB

        print('Plotting...')

        plt.figure('blur')
        imsave('_blur.JPG', original_img)
        # plt.imshow(original_img)

        plt.figure('predict')
        imsave('_predict.JPG', predict_img)
        # plt.imshow(predict_img)
        # plt.imshow(predict_img_patches[0][..., ::-1])

        # sharp_img = skimage.img_as_float(imread('/home/doleinik/SharpeningPhoto/quality_ImageNet/test_500/images/100_sh.JPEG'))[:128, :128]
        # sharp_img = sharp_img[..., ::-1]
        # sharp_img = np.transpose(sharp_img, (2, 0, 1))
        # plt.figure('sharp')
        # plt_img(sharp_img)

        # plt.show()
    else:
        # from lmdb
        lmdb_path = '/home/doleinik/SharpeningPhoto/lmdb/'
        # paths = [lmdb_path + 'train_blur_lmdb_128', lmdb_path + 'train_sharp_lmdb_128']
        paths = [lmdb_path + 'val_blur_lmdb_128', lmdb_path + 'val_sharp_lmdb_128']

        list_ids = [567, 345, 2344]
        ids = ['{:08}'.format(i) for i in list_ids]
        blur_data, sharp_data = SP_model.get_data_from_keys(paths, ids)

        predict_data_1 = model.predict(blur_data)
        # predict_data_2 = model.predict(predict_data_1)

        save_dir_imgs = '/home/doleinik/trained_models_SharpeningPhoto/imgs/'
        if not os.path.isdir(save_dir_imgs):
            os.makedirs(save_dir_imgs)

        for i in range(len(ids)):
            name = ids[i] + '_' + train_name + '_blur'
            plt.figure(name)
            plt.title(name)
            img = plt_img(blur_data[i])
            imsave(save_dir_imgs + name + '.png', img)
            plt.close()

            name = ids[i] + '_' + train_name + '_sharp'
            plt.figure(name)
            plt.title(name)
            img = plt_img(sharp_data[i])
            imsave(save_dir_imgs + name + '.png', img)
            plt.close()

            name = ids[i] + '_' + train_name + '_pred'
            plt.figure(name)
            plt.title(name)
            img = plt_img(predict_data_1[i])
            imsave(save_dir_imgs + name + '.png', img)
            plt.close()

            # name = ids[i] + '_' + train_name + '_pred_pred'
            # plt.figure(name)
            # plt.title(name)
            # img = plt_img(predict_data_2[i])
            # imsave(save_dir_imgs + name + '.png', img)
            # plt.close()

        # plt.show()


if __name__ == '__main__':

    train_names = [
        ['mean_squared_error_lr_0.001',39500],
        ['mean_squared_error_lr_0.00002',500],
        ['laplacian_gray_loss',37500],
        ['sub_loss',39500],
        ['clip_laplacian_color_loss',500],
        ['mean_squared_error_lr_0.001_w_BN',4500]
    ]
    for tr in train_names:
        print('--> step ' + tr[0])
        graph_metrics(tr[0])
        evaluate(tr[0], tr[1])
    print('>------end------<')