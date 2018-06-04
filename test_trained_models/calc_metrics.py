# -*- coding: utf-8 -*-

from __future__ import print_function

import cv2

ksize_laplacian = 3
ksize_median = 3

def calc_of_laplacian(gray_img, type_of_calc):
    median = cv2.medianBlur(gray_img, ksize_median)
    laplacian = cv2.Laplacian(median, -1, ksize=ksize_laplacian)
    if type_of_calc == 'std':
        return laplacian.std()
    elif type_of_calc == 'mean':
        return laplacian.mean()

if __name__ == '__main__':
    list_ids = [27,
                68,
                561]
    # models_postfix = ['_l15_mean_squared_error_lr_0.001_iter_4800_pred.png',
    #                   '_mean_squared_error_lr_0.001_b1_iter_140000_pred.png',
    #                   '_mean_squared_error_lr_0.001_iter_5600_pred.png',
    #                   '_mean_squared_error_lr_0.001_w_BN_kernel_init_iter_5000_pred.png',
    #                   '_mean_squared_error_lr_0.001_w_relu_iter_5200_pred.png',
    #                   '_mean_squared_error_lr_0.00002_iter_5000_pred.png',
    #                   '_spn_mean_squared_error_lr_0.001_iter_5000_pred.png',
    #                   '_sub_loss_iter_5000_pred.png']

    train_names = [
        ['mean_squared_error_lr_0.001_w_relu', 6200],
        ['mean_squared_error_lr_0.001_w_relu', 13600],
        ['mean_squared_error_lr_0.001_w_BN_kernel_init', 5000],
        ['spn_mean_squared_error_lr_0.001', 5000],
        ['l15_mean_squared_error_lr_0.001', 4800]
    ]
    models_postfix = map(lambda x:'_'+x[0]+'_iter_'+str(x[1])+'_pred.png', train_names)
    img_path = '/Users/dmitryoleynik/PycharmProjects/SharpeningPhoto/test_trained_models/final'

    metrics_mean = []
    metrics_std = []
    metrics_PSNR = []
    metrics_mean.append(['img_id', 'sharp', 'blur'])
    metrics_std.append(['img_id', 'sharp', 'blur'])
    metrics_PSNR.append(['img_id', 'blur'])
    for mp in models_postfix:
        metrics_mean[0].append(mp)
        metrics_std[0].append(mp)
        metrics_PSNR[0].append(mp)

    for id in list_ids:
        l_mean = []
        l_mean.append(id)
        l_std = []
        l_std.append(id)
        l_PSNR = []
        l_PSNR.append(id)

        sharp = cv2.imread(img_path + '/' + str(id) + '_sharp.png', 0)
        blur = cv2.imread(img_path + '/' + str(id) + '_blur.png', 0)
        # sharp = cv2.cvtColor(cv2.imread(img_path + '/' + str(id) + '_sharp.png'), cv2.COLOR_BGR2YCrCb)[..., 0]
        # blur = cv2.cvtColor(cv2.imread(img_path + '/' + str(id) + '_blur.png'), cv2.COLOR_BGR2YCrCb)[..., 0]


        sharp_mean = calc_of_laplacian(sharp, 'mean')
        l_mean.append(sharp_mean)
        sharp_std = calc_of_laplacian(sharp, 'std')
        l_std.append(sharp_std)

        blur_mean = calc_of_laplacian(blur, 'mean')
        l_mean.append(blur_mean)
        blur_std = calc_of_laplacian(blur, 'std')
        l_std.append(blur_std)
        blur_PSNR = cv2.PSNR(sharp, blur)
        l_PSNR.append(blur_PSNR)


        for mp in models_postfix:
            pred_img = cv2.imread(img_path + '/' + str(id) + mp, 0)
            pred_mean = calc_of_laplacian(pred_img, 'mean')
            l_mean.append(pred_mean)
            pred_std = calc_of_laplacian(pred_img, 'std')
            l_std.append(pred_std)
            pred_PSNR = cv2.PSNR(sharp, pred_img)
            l_PSNR.append(pred_PSNR)

        metrics_mean.append(l_mean)
        metrics_std.append(l_std)
        metrics_PSNR.append(l_PSNR)

    f_mean = open('metrics_mean.csv','w')
    f_mean.write(','.join(metrics_mean[0]) + '\n')
    for i in metrics_mean[1:]:
        f_mean.write(','.join('{:.2f}'.format(x) for x in i) + '\n')

    f_std = open('metrics_std.csv','w')
    f_std.write(','.join(metrics_std[0]) + '\n')
    for i in metrics_std[1:]:
        f_std.write(','.join('{:.2f}'.format(x) for x in i) + '\n')

    f_PSNR = open('metrics_PSNR.csv','w')
    f_PSNR.write(','.join(metrics_PSNR[0]) + '\n')
    for i in metrics_PSNR[1:]:
        f_PSNR.write(','.join('{:.2f}'.format(x) for x in i) + '\n')