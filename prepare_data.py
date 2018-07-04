# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use('Agg')
from scipy import interpolate
import os
import glob
import h5py


'''
ГЕНЕРАЦИЯ ШУМА
'''

def bool_rand():
    return np.random.choice([True, False])

# интервал начального значения СКО
interval = [1, 7]
def gen_std():
    # std_rand = np.random.randint(interval[0], interval[1], dtype=np.uint8)  # рандом начального значения для всех каналов
    std_rand = np.random.randint(interval[0], interval[1], dtype=np.uint8)  # рандом начального значения для всех каналов
    # std_rand = std_rand + np.random.randint(0, 7, size=3, dtype=np.uint8)  # рандом отклоения каждого RGB канала
    std_rand = std_rand + np.random.randint(0, 7, dtype=np.uint8)  # рандом отклоения каждого RGB канала
    return std_rand


def gen_noise(img, sigma_rand, resize_scale):
    '''
    Генерация шума в соответствии с размером изображения.
    Заполнение нормально распределенными случайными числами.
    '''

    # h, w, c = img.shape
    h, w = img.shape

    # нормальное распределение (mu, sigma(std) - варируемые величины)
    # mu = np.zeros(c, np.uint8)
    mu = np.zeros(1, np.uint8)
    # sigma = np.ones(c, np.uint8) * sigma_rand
    sigma = np.ones(1, np.uint8) * sigma_rand

    # noise = np.zeros((h, w, c), np.uint8)
    noise = np.zeros((h, w), np.uint8)

    # заполение шума нормально распределенными случайными числами
    cv2.randn(noise, mu, sigma)          # |\__
    # cv2.randn(noise, mu * 127, sigma)  # |_/\_

    # заполение шума равномерно распределенными случайными числами
    # cv2.randu(noise, mu, sigma)

    noise = cv2.resize(noise, None, fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_CUBIC)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)

    return noise


def add_noise(img):
    std_rand = gen_std()
    resize_scale = 0.25 * np.random.random() + 0.75
    print(std_rand)
    print(resize_scale)
    noise_to_add = gen_noise(img, std_rand, resize_scale)
    noise_to_subtract = gen_noise(img, std_rand, resize_scale)

    # добавление шума к размытым фото
    if bool_rand():
        img_with_noise = cv2.add(img, noise_to_add)
        img_with_noise = cv2.subtract(img_with_noise, noise_to_subtract)
    else:
        img_with_noise = cv2.subtract(img, noise_to_subtract)
        img_with_noise = cv2.add(img_with_noise, noise_to_add)

    return img_with_noise


'''
ГЕНЕРАЦИЯ РАЗМЫТИЯ
'''

fb_kernel_sizes = np.arange(5, 52, 2)
def do_focus_blur(input_img):
    '''
    Фокусное размытие
    :param input_img:
    :return:
    '''
    ksize = np.random.choice(fb_kernel_sizes)
    return cv2.GaussianBlur(input_img, (ksize, ksize), 0)


mb_img_shape = (200, 200)
mb_kernel_sizes = np.arange(1, 52, 2)
def do_motion_blur(input_img):
    '''
    размытие движения
    :param input_img:
    :return:
    '''
    n = np.random.randint(2, 5)  # колечество точек
    if n == 2:
        k = 1  # степень полинома при интерполяции
    elif n == 3:
        k = 2
    else:
        k = 3

    # генерация сплайна
    points = np.random.randint(50, 150, (2, n))
    tck, u = interpolate.splprep(points, s=0, k=k)
    x, y = interpolate.splev(np.linspace(0, 1, 100), tck, der=0)
    spline_points = np.array([x, y], np.int32).T

    # добавление сплайна на изображение
    black_img = np.zeros(mb_img_shape, np.uint8)

    thickness = np.random.randint(1, 30)
    cv2.polylines(black_img, [spline_points], False, 255, thickness, cv2.LINE_AA)

    # размытие изображения
    ksize = tuple(np.random.choice(mb_kernel_sizes, 2))
    blured_img = cv2.GaussianBlur(black_img, ksize, 0)

    # получение границ сплайна
    indices = np.nonzero(blured_img)
    y_min, x_min = np.amin(indices, axis=1)
    y_max, x_max = np.amax(indices, axis=1)

    y_max = y_max if (y_max - y_min) % 2 == 1 else (y_max + 1)
    x_max = x_max if (x_max - x_min) % 2 == 1 else (x_max + 1)

    # обрезка сплайна
    croped_img = blured_img[y_min:y_max, x_min:x_max]
    final_img = croped_img.astype(np.float32) / np.sum(croped_img)

    # сохранение ядра размытия движения
    final_img_name = 'croped n:' + str(n) + ' k:' + str(k) + ' thickness:' + str(thickness) + ' ksize:' + str(ksize)
    cv2.imwrite('kernels/' + final_img_name + '.png', croped_img)

    return cv2.filter2D(input_img, -1, final_img)


'''
ФОРМИРОВАНИЕ ДАННЫХ
'''

block_step = 32
block_size = 64

def create_hdf5(imgs_dir, hdf5_path):

    print('Обработка директории: ' + imgs_dir)

    f = h5py.File(hdf5_path, 'w')
    dset_data = f.create_dataset('data',
                                 shape=(0, block_size, block_size, 1),
                                 dtype=np.float32,
                                 maxshape=(None, block_size, block_size, 1),
                                 chunks=(1024*10, block_size, block_size, 1))
    dset_label = f.create_dataset('label',
                                  shape=(0, block_size, block_size, 1),
                                  dtype=np.float32,
                                  maxshape=(None, block_size, block_size, 1),
                                  chunks=(1024*10, block_size, block_size, 1))

    for img_path in sorted(glob.glob(os.path.join(imgs_dir, '*.*'))):
        print(img_path)

        img = cv2.imread(img_path)
        Y = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[..., 0]
        height, width = Y.shape

        if height >= block_size:
            h_num = int((height - block_size) / block_step) + 1
        else:
            h_num = 0

        if width >= block_size:
            w_num = int((width - block_size) / block_step) + 1
        else:
            w_num = 0

        data = []
        label = []

        Y_fb = do_focus_blur(Y)
        Y_mb = do_motion_blur(Y)

        Y_fb = add_noise(Y_fb)
        Y_mb = add_noise(Y_mb)

        for Y_blured in [Y_fb, Y_mb]:

            for i in range(h_num):
                for j in range(w_num):
                    y = i * block_step
                    x = j * block_step

                    data_patch = Y_blured[y:y+block_size, x:x+block_size]
                    label_patch = Y[y:y+block_size, x:x+block_size]

                    data_patch = data_patch / 255.
                    label_patch = label_patch / 255.

                    data_patch.resize((block_size, block_size, 1))
                    label_patch.resize((block_size, block_size, 1))

                    data.append(data_patch)
                    label.append(label_patch)
        data = np.array(data, np.float32)
        label = np.array(label, np.float32)
        print(data.shape)

        # запись в БД
        dset_idx_start = dset_data.shape[0]
        dset_idx_stop = dset_idx_start + data.shape[0]

        dset_data.resize(dset_idx_stop, axis=0)
        dset_label.resize(dset_idx_stop, axis=0)

        dset_data[dset_idx_start:dset_idx_stop] = data
        dset_label[dset_idx_start:dset_idx_stop] = label
        print(dset_data.shape)

        f.flush()

    f.close()
    print('Создана база: ' + hdf5_path + '\n')


def read_hdf5(filename):
    f = h5py.File(filename, 'r')
    data = f['data'][:]
    label = f['label'][:]
    f.close()
    return data, label


if __name__ == '__main__':

    # создание обучающих баз данных
    create_hdf5('dataset/train', 'dataset/train.h5')
    create_hdf5('dataset/val', 'dataset/val.h5')
    create_hdf5('dataset/test', 'dataset/test.h5')


    # чтение базы данных
    # train_data, train_label = read_hdf5('dataset/train.h5')
    # val_data, val_label = read_hdf5('dataset/val.h5')
    # test_data, test_label = read_hdf5('dataset/test.h5')

    f = h5py.File('dataset/train.h5', 'r')

    cv2.imshow('data', f['data'][0].reshape(block_size, block_size))
    cv2.imshow('label', f['label'][0].reshape(block_size, block_size))
    f.close()
    cv2.waitKey()
