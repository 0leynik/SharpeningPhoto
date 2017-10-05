# -*- coding: utf-8 -*-
import numpy as np
import glob
import os
import cv2
import matplotlib.pyplot as plt
import math

bins = range(0, 256)

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


def noise_testing_on_gray_background():
    '''
    Имитация добавления шума на сером фоне (12)
    шум генерируется с помощью нормального распределения

    ПРОТЕСТИРОВАТЬ НА РЕАЛЬНОЙ ФОТКЕ с шумом где ноут
    '''
    print os.listdir('.')
    # img_path = 'gray.jpg'
    img_path = '8.JPEG'
    # img_path = '6.JPG'
    img = cv2.imread(img_path)
    # img = cv2.GaussianBlur(img, (15, 15), 0)
    # img = cv2.GaussianBlur(img, (43, 43), 0)
    # img = cv2.GaussianBlur(img, (67, 67), 0)
    h, w, c = img.shape

    '''
    равномерное распределение
    '''
    # noise = np.zeros((h, w, 3), np.uint8)
    # cv2.randu(noise, np.zeros(3), np.ones(3) * 256)
    # plt.figure('uniform distribution')
    # plt.hist(noise.ravel(), bins)
    # cv2.imshow('uniform noise', noise)

    '''
    нормальное распределение (mu, sigma() - варируемые величины)
    '''
    interval = [2, 7]

    resize = False
    fx = 0.75
    fy = 0.75

    mu = np.zeros(c)
    sigma_rand = np.random.randint(interval[0], interval[1])  # рандом начального значения для всех каналов
    sigma_rand = sigma_rand + np.random.randint(0, 7, size=3)  # рандом отклоения каждого из каналов
    # sigma_rand = np.random.randint(16,17, size=3)  # рандом отклоения каждого из каналов
    print sigma_rand
    sigma = np.ones(c) * sigma_rand
    print sigma

    # шум 1
    noise_1 = np.zeros((h, w, c), np.uint8)
    cv2.randn(noise_1, mu, sigma)        # |\__
    # cv2.randn(noise, mu * 127, sigma)  # |_/\_
    if resize:
        noise_1 = cv2.resize(noise_1, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
        noise_1 = cv2.resize(noise_1, (w, h), interpolation=cv2.INTER_CUBIC)
    # plot_hist('noise_1', noise_1)
    # cv2.imshow('noise_1', noise_1)

    # шум 2
    noise_2 = np.zeros((h, w, c), np.uint8)
    cv2.randn(noise_2, mu, sigma)        # |\__
    # cv2.randn(noise, mu * 127, sigma)  # |_/\_
    if resize:
        noise_2 = cv2.resize(noise_2, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
        noise_2 = cv2.resize(noise_2, (w,h), interpolation=cv2.INTER_CUBIC)
    # plot_hist('noise_2', noise_2)
    # cv2.imshow('noise_2', noise_2)


    '''
    ШУМ НА СЕРОМ ФОНЕ
    '''
    # серый фон
    gray = np.ones((h, w, c), np.uint8) * 128
    # plot_hist('gray', gray)

    gray_1 = cv2.add(gray, noise_1)
    gray_1 = cv2.subtract(gray_1, noise_2)
    plot_hist('add+sub', gray_1)
    cv2.imshow('add+sub', gray_1)

    gray_2 = cv2.subtract(gray, noise_1)
    gray_2 = cv2.add(gray_2, noise_2)
    # plot_hist('sub+add', gray_2)
    # cv2.imshow('sub+add', gray_2)

    '''
    ШУМ НА ЦВЕТНЫХ ФОТО
    '''
    # plot_hist('img', img)

    img_1 = cv2.add(img, noise_1)
    img_1 = cv2.subtract(img_1, noise_2)
    # plot_hist('add+sub img', img_1)
    cv2.imshow('add+sub img', img_1)

    img_2 = cv2.subtract(img, noise_1)
    img_2 = cv2.add(img_2, noise_2)
    # plot_hist('sub+add img', img_2)
    # cv2.imshow('sub+add img', img_2)


    plt.show()
    cv2.waitKey()


if __name__ == '__main__':
    noise_testing_on_gray_background()