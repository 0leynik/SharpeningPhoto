# -*- coding: utf-8 -*-

from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import glob
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import cv2
from skimage.io import imread, imshow, imsave
import skimage
import keras
import keras.backend as K
import SP_model


def get_models_paths(train_dir):
    return glob.glob(train_dir+'/saved_models/iter_*.h5')

def graph_metrics(train_name, savefig=True, show=False):
    print('\n--> metrics \"' + train_name + '\"')

    model_path = work_dir + '/' + train_name + '/metrics.csv'

    graphs_savedir = work_dir + '/graphs'
    if not os.path.isdir(graphs_savedir):
        os.makedirs(graphs_savedir)

    metrics = np.loadtxt(model_path, delimiter=',')

    loss_name = train_name + ' loss'
    plt.figure(loss_name)
    # plt.title(loss_name)
    plt.plot(metrics[:, 1])
    if metrics.shape[1]==3:
        plt.plot(metrics[:, 2])
        plt.legend(['train', 'val'])
    # plt.ylabel('loss')
    # plt.xlabel('iter')
    plt.grid(True, linestyle='--')
    plt.yticks(np.linspace(0., 0.05, 11))
    plt.ylim(0,0.05)
    plt.xticks(np.linspace(0, 5000, 11))
    plt.xlim(0, 5000)
    if savefig:
        plt.savefig(graphs_savedir + '/' + loss_name + '.png')
    plt.close()

    # acc_name = 'acc ' + train_name
    # plt.figure(acc_name)
    # plt.title(acc_name)
    # plt.plot(metrics[:, 2])
    # plt.ylabel('acc')
    # plt.xlabel('iter')
    # plt.grid(True, linestyle='--')
    # # plt.yticks(np.linspace(0., 0.2, 10))
    # # plt.ylim(0., 0.2)
    # if savefig:
    #     plt.savefig(save_dir_graphs + acc_name + '.png')
    # # plt.close()

    # if show:
    #     plt.show()


def plt_img(data):
    img = np.transpose(data, (1, 2, 0)) # CxHxW -> HxWxC
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[:, :, ::-1] # BGR -> RGB
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
        # print(height.shape)
        for width in np.split(height, w_count, axis=1):
            # print(width.shape)
            patches[i] = width
            i += 1

    return patches

def get_img_from_patches(patches, img):
    h, w, c = img.shape
    b_count = patches.shape[0]
    # print(patches.shape)
    extended_img = np.zeros((h_count * h_step, w_count * w_step, c), dtype=np.float32)
    k = 0
    for i in range(h_count):
        for j in range(w_count):
            extended_img[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step] = patches[k]
            k += 1

    return extended_img[:h, :w]

def evaluate(load_imgs_from_db, train_name, iter_num=None):
    print('\n--> evaluate \"' + train_name + '\"')
    custom_objects = {
        'laplacian_gray_loss': SP_model.laplacian_gray_loss,
        'sub_loss': SP_model.sub_loss,
        'clip_laplacian_color_loss': SP_model.clip_laplacian_color_loss
    }


    imgs_savedir = work_dir + '/imgs'
    if not os.path.isdir(imgs_savedir):
        os.makedirs(imgs_savedir)

    if load_imgs_from_db:
        # load db images
        list_ids = [27,
                    42,
                    68,
                    84,
                    138,
                    176,
                    179,
                    201,
                    212,
                    284,
                    561,
                    620,
                    650,
                    791,
                    841,
                    922,
                    934,
                    937,
                    956,
                    959]
        list_ids = [1127,1152,1153,1178,1348,1529,1558,1574,2060,2097,2192,2237,2735,2768]
        ids = ['{:08}'.format(i) for i in list_ids]
        if on_P:
            lmdb_path = '/home/cudauser/SharpeningPhoto/lmdb'
        else:
            lmdb_path = '/home/doleinik/SharpeningPhoto/lmdb'
        paths = [lmdb_path + '/test_blur_lmdb_128', lmdb_path + '/test_sharp_lmdb_128']
        blur_data, sharp_data = SP_model.get_data_from_keys(paths, ids)
    else:
        # img_dir = '/Users/dmitryoleynik/PycharmProjects/SharpeningPhoto/models/imgs'
        # img_names = ['sat1.jpg','sat6.jpg']

        img_dir = '/Users/dmitryoleynik/PycharmProjects/SharpeningPhoto/models/imgs/images_sereja_2.0'
        # img_names = ['DbdoGJUXkAE1KUf.jpg']
        # img_names = ['Screenshot_63.jpg']
        img_names = ['rentgen-legkih.jpg']
        img_names = ['8755c957df48bf75ed3e9017296fb3a7.jpg',
'd886c65abcdba1ff74fe8b61ecd98d61.jpg',
'e5335a8a82444a06aae7c6e4b1d6ea0d.jpg',
'eff36d77a583b46e461c12a3c0037063.png']
        # img_names = ['q9koKk6tURM.jpg',
        #              'wall_input_small.png',
        #              'wall_x4_bicubic.jpg',
        #              '1blur.JPG',
        #              '4.png',
        #              '10.png',
        #              '18.png',
        #              '32.png',
        #              '35.png',
        #              '54.png',
        #              '72.png',
        #              '83.png',
        #              '144.png',
        #              '155.png',
        #              '159.png',
        #              '2288.png',
        #              '2471.png',
        #              '2486.png',
        #              '2504.png',
        #              '2556.png',
        #              '2584.png',
        #              '2647.png',
        #              '2652.png',
        #              '2708.png',
        #              '2711.png',
        #              '2731.png',
        #              '2786.png',
        #              '2912.png',
        #              '2923.png',
        #              '3024.png',
        #              '3061.png',
        #              '3134.png',
        #              '3156.png',
        #              '3221.png',
        #              '3237.png',
        #              '3249.png',
        #              '3274.png',
        #              '3702.png',
        #              '3733.png',
        #              '3782.png',
        #              '3802.png',
        #              '3825.png',
        #              '3893.png',
        #              '4081.png',
        #              '4112.png',
        #              '4144.png',
        #              '4155.png',
        #              '4172.png',
        #              '4208.png',
        #              '4287.png',
        #              '4292.png',
        #              '4298.png',
        #              '4354.png',
        #              '4361.png',
        #              '4606.png',
        #              '6044.png',
        #              '6097.png',
        #              '6142.png',
        #              '6143.png',
        #              '6146.png',
        #              '6192.png',
        #              '6240.png',
        #              '6290.png',
        #              '6311.png',
        #              '6328.png']

    if iter_num is None:
        models_paths = get_models_paths(work_dir+'/'+train_name)
    else:
        models_paths = [work_dir + '/' + train_name + '/saved_models/iter_' + str(iter_num) + '.h5']

    for model_path in models_paths:

        iter_name = os.path.splitext(os.path.basename(model_path))[0]
        print(iter_name)

        if load_imgs_from_db:
            model = keras.models.load_model(model_path, custom_objects=custom_objects)
            predict_data = model.predict(blur_data)

            for i in range(len(list_ids)):
                print(i)
                img_savepath = imgs_savedir + '/' + str(list_ids[i]) + '_' + train_name + '_' + iter_name
                mplimg.imsave(img_savepath + '_blur.png', plt_img(blur_data[i]))
                mplimg.imsave(img_savepath + '_sharp.png', plt_img(sharp_data[i]))
                mplimg.imsave(img_savepath + '_pred.png', plt_img(predict_data[i]))
        else:
            for img_name in img_names:
                print(img_name)
                img_path = img_dir + '/' + img_name

                # original_img = skimage.img_as_float(imread(img_path)[..., :3])
                # # print('shape original: ' + str(original_img.shape))
                #
                # img = original_img[:, :, ::-1]  # RGB -> BGR
                # img_patches = get_patches_from_img(img)
                # img_patches = np.array(map(lambda i: np.transpose(i, (2, 0, 1)), img_patches), np.float32)  # HxWxC -> CxHxW
                # # print('shape patches: ' + str(img_patches.shape))
                #
                # model = keras.models.load_model(model_path, custom_objects=custom_objects)
                # predict_img_patches = model.predict(img_patches)
                #
                # predict_img_patches = np.array(map(lambda i: np.transpose(i, (1, 2, 0)), predict_img_patches), np.float32)  # HxWxC -> CxHxW
                # predict_img = get_img_from_patches(predict_img_patches, original_img)
                # predict_img = predict_img[:, :, ::-1]  # BGR -> RGB
                #
                # img_savepath = imgs_savedir + '/' + os.path.splitext(img_name)[0] + '_' + train_name + '_' + iter_name
                # mplimg.imsave(img_savepath + '_blur.png', original_img)
                # mplimg.imsave(img_savepath + '_pred.png', predict_img)


                original_img = (cv2.imread(img_path)/255.)[:, :, ::-1]
                print('shape original: ' + str(original_img.shape))

                img = original_img[:, :, ::-1]  # RGB -> BGR
                h, w, c = img.shape
                img = np.transpose(img, (2, 0, 1))  # HxWxC -> CxHxW

                img_to_pred = np.empty((1, c, 128 * (h/128 + 1), 128 * (w/128 + 1)))
                print('shape pred: ' + str(img_to_pred.shape))
                img_to_pred[0, :, :h, :w] = img

                model = SP_model.get_unet_128_w_relu()
                model.summary()

                # print('Metrics: ' + str(model.metrics_names))
                # model.compile(optimizer='adam', loss='mean_squared_error')
                model.load_weights(model_path)
                predict_img = model.predict(img_to_pred)


                predict_img = np.transpose(predict_img[0, :, :h, :w] , (1, 2, 0))  # HxWxC -> CxHxW
                predict_img = predict_img[:, :, ::-1]  # BGR -> RGB

                img_savepath = imgs_savedir + '/' + os.path.splitext(img_name)[0] + '_' + train_name + '_' + iter_name
                mplimg.imsave(img_savepath + '_blur.png', original_img)
                mplimg.imsave(img_savepath + '_pred.png', predict_img)


mpl.rcParams['figure.figsize'] = [8.4, 4.8]
mpl.rcParams['figure.dpi'] = 500
mpl.rcParams['lines.linewidth'] = 0.7
mpl.rcParams['axes.linewidth'] = 0.3

on_cluster = False
on_P = False
if on_cluster:
    if on_P:
        work_dir = '/home/cudauser/trained_models'
    else:
        work_dir = '/home/doleinik/trained_models'
    load_imgs_from_db = True
else:
    work_dir = '/Users/dmitryoleynik/PycharmProjects/SharpeningPhoto/models/trained_models'
    load_imgs_from_db = False

if __name__ == '__main__':

    # train_names = [
    #     ['mean_squared_error_lr_0.001_w_relu',6200],
    #     ['mean_squared_error_lr_0.001_w_relu',13600],
    #     ['mean_squared_error_lr_0.001_w_BN_kernel_init',5000],
    #     ['spn_mean_squared_error_lr_0.001',5000],
    #     ['l15_mean_squared_error_lr_0.001',4800]
    # ]
    train_names = [
        # ['mean_squared_error_lr_0.001_w_relu',6200],
        ['mean_squared_error_lr_0.001_w_relu',13600]
    ]

    for tr in train_names:
        # graph_metrics(train_name=tr[0])

        if len(tr) == 1:
            evaluate(load_imgs_from_db, train_name=tr[0])
        elif len(tr) == 2:
            evaluate(load_imgs_from_db, train_name=tr[0], iter_num=tr[1])

    print('\n--> end')
