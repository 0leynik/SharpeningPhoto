# -*- coding: utf-8 -*-
import numpy as np
import glob
import os
import cv2
import matplotlib.pyplot as plt

ksize_laplacian = 3
ksize_median = 3

'''
параметры шума
'''
# интервал начального значения СКО
interval = [1, 7]


bins = range(0, 256)

def do_img_for_visio():
    for i in range(1, 12):
        fileName = str(i) + '.JPEG'
        gray_img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
        resize_img = cv2.resize(gray_img, (512, 512), interpolation=cv2.INTER_CUBIC)
        # median = resize_img
        median = cv2.medianBlur(resize_img, ksize_median)
        laplacian = cv2.Laplacian(median, -1, ksize=ksize_laplacian)
        print laplacian.std()

        cv2.imwrite('gray_' + fileName, gray_img)
        cv2.imwrite('resize_' + fileName, resize_img)
        cv2.imwrite('median_' + fileName, median)
        cv2.imwrite('laplacian_' + fileName, laplacian)

def plot_hist(title,img):

    plt.figure(title)

    # x = range(0, 256, 16)
    # plt.xticks(x)  # значения на оси

    plt.hist(img[:,:,0].ravel(), bins, histtype = 'stepfilled', color='b', edgecolor='b', alpha = 0.5)
    plt.hist(img[:,:,1].ravel(), bins, histtype = 'stepfilled', color='g', edgecolor='g', alpha = 0.5)
    plt.hist(img[:,:,2].ravel(), bins, histtype = 'stepfilled', color='r', edgecolor='r', alpha = 0.5)
    plt.hist(img.ravel(), bins, histtype = 'stepfilled', color='k', alpha = 0.5)
    # plt.hist(img.ravel(), bins, histtype = 'bar', color='k', edgecolor='r')

    # plt.xlim(80, 176)  # интервал оси
    # plt.ylim(0, 5000)  # интервал оси


def gen_noise(img, sigma_rand, resize_scale):
    '''
    Генерация шума в соответствии с размером изображения.
    Заполнение нормально распределенными случайными числами.
    '''
    h, w, c = img.shape

    # нормальное распределение (mu, sigma(std) - варируемые величины)
    mu = np.zeros(c, np.uint8)
    sigma = np.ones(c, np.uint8) * sigma_rand

    noise = np.zeros((h, w, c), np.uint8)

    # заполение шума нормально распределенными случайными числами
    cv2.randn(noise, mu, sigma)          # |\__
    # cv2.randn(noise, mu * 127, sigma)  # |_/\_

    # заполение шума равномерно распределенными случайными числами
    # cv2.randu(noise, mu, sigma)

    noise = cv2.resize(noise, None, fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_CUBIC)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)

    return noise


def test_noise():
    dir_to_save = 'test noise interval=['+str(interval[0])+', '+str(interval[1])+']'
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)

    ext = '.JPEG'
    for i in range(1, 12):
        print i
        fileName = str(i) + ext

        # исходное и размытое изображение
        img = cv2.imread(fileName)
        # blured_img = cv2.filter2D(img, -1, numpy.eye(21)/21)
        blured_img = cv2.GaussianBlur(img, (21, 21), 0)

        cv2.imwrite(dir_to_save + '/' + str(i) + ext, img)
        cv2.imwrite(dir_to_save + '/' + str(i) + ' blured' + ext, blured_img)

        '''поиск шума при помощи медианного фильтра'''
        # # изображения без шумов
        # nlmean = cv2.fastNlMeansDenoisingColored(img)
        # median = cv2.medianBlur(img, ksize_median)
        # cv2.imwrite(dir_to_save + '/' + str(i) + ' rem noise nlmean' + ext, nlmean)
        # cv2.imwrite(dir_to_save + '/' + str(i) + ' rem noise median' + ext, median)
        #
        # # только шум
        # noise_nlmean = cv2.subtract(img, nlmean)
        # noise_median = cv2.subtract(img, median)
        # cv2.imwrite(dir_to_save + '/' + str(i) + ' noise img-nlmean' + ext, noise_nlmean)
        # cv2.imwrite(dir_to_save + '/' + str(i) + ' noise img-median' + ext, noise_median)
        # noise_nlmean = cv2.subtract(nlmean, img)
        # noise_median = cv2.subtract(median, img)
        # cv2.imwrite(dir_to_save + '/' + str(i) + ' noise nlmean-img' + ext, noise_nlmean)
        # cv2.imwrite(dir_to_save + '/' + str(i) + ' noise median-img' + ext, noise_median)
        #
        # # шум с порогом среднего значения по всем каналам
        # print noise_nlmean.mean()
        # plt.figure('noise_nlmean')
        # plt.hist(noise_nlmean.ravel(), bins)
        # plt.savefig(dir_to_save + '/' + str(i) + ' noise_nlmean.PNG')
        # plt.clf()
        # print noise_median.mean()
        # plt.figure('noise_median')
        # plt.hist(noise_median.ravel(), bins)
        # plt.savefig(dir_to_save + '/' + str(i) + ' noise_median.PNG')
        # plt.clf()

        '''параметры амплитуды'''
        # генерация СКО (рандомное значение сигмы по каждому из RGB канала)
        sigma_rand = np.random.randint(interval[0], interval[1], dtype=np.uint8)  # рандом начального значения для всех каналов
        sigma_rand = sigma_rand + np.random.randint(0, 7, size=3, dtype=np.uint8)  # рандом отклоения каждого из каналов
        # sigma_rand = np.random.randint(14, 15, size=3)  # рандом отклоения каждого из каналов
        print sigma_rand
        '''параметры частоты'''
        resize_scale = 0.25 * np.random.random() + 0.75

        '''генерация шума'''
        noise_to_add = gen_noise(img, sigma_rand, resize_scale)
        noise_to_subtract = gen_noise(img, sigma_rand, resize_scale)

        # plot_hist('noise_to_add', noise_to_add)
        # plt.savefig(dir_to_save + '/' + str(i) + ' noise_to_add HIST.PNG')
        plt.clf()
        # plot_hist('noise_to_subtract', noise_to_subtract)
        # plt.savefig(dir_to_save + '/' + str(i) + ' noise_to_subtract HIST.PNG')
        plt.clf()


        # cv2.imwrite(dir_to_save + '/' + str(i) + ' noise to add' + ext, noise_to_add)
        # cv2.imwrite(dir_to_save + '/' + str(i) + ' noise to subtract' + ext, noise_to_subtract)
        # cv2.imwrite(dir_to_save + '/' + str(i) + ' noise to add (bitwise_not)' + ext, cv2.bitwise_not(noise_to_add))
        # cv2.imwrite(dir_to_save + '/' + str(i) + ' noise to subtract (bitwise_not)' + ext, cv2.bitwise_not(noise_to_subtract))


        # добавление шума к размытым фото
        if np.random.choice([True, False]):
            blured_add_noise = cv2.add(blured_img, noise_to_add)
            blured_add_noise = cv2.subtract(blured_add_noise, noise_to_subtract)
            cv2.imwrite(dir_to_save + '/' + str(i) + ' blured noise add+subtract ' + str(sigma_rand) + ext, blured_add_noise)
        else:
            blured_add_noise = cv2.subtract(blured_img, noise_to_subtract)
            blured_add_noise = cv2.add(blured_add_noise, noise_to_add)
            cv2.imwrite(dir_to_save + '/' + str(i) + ' blured noise subtract+add ' + str(sigma_rand) + ext, blured_add_noise)


if __name__ == '__main__':
    # do_img_for_visio()
    test_noise()
