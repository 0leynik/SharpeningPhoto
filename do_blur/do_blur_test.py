# -*- coding: utf-8 -*-
import os
import glob
import cv2
import numpy as np


def bool_rand():
    return np.random.choice([True, False])


'''
ГЕНЕРАЦИЯ ШУМА
'''

# интервал начального значения СКО
interval = [1, 7]
def gen_std():
    std_rand = np.random.randint(interval[0], interval[1], dtype=np.uint8)  # рандом начального значения для всех каналов
    std_rand = std_rand + np.random.randint(0, 7, size=3, dtype=np.uint8)  # рандом отклоения каждого RGB канала
    # std_rand = np.random.randint(14, 15, size=3)  # рандом отклоения каждого из каналов
    return std_rand


def gen_resize_scale():
    return 0.25 * np.random.random() + 0.75


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


def add_noise(img):
    std_rand = gen_std()
    resize_scale = gen_resize_scale()
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

blur_type = 0 # 1(mb) 2(fb) 3(mfb)
kernel_bounds_mb  = [13.0/4320, 69.0/4320]
kernel_bounds_fb  = [19.0/4320, 71.0/4320]
kernel_bounds_mfb = [11.0/4320, 37.0/4320]

# 15*500/4320=1,736111111111111
# 71*500/4320=8,217592592592593

def gen_kernel_size(img): # определение размера ядра свертки
    height, width, channels = img.shape
    img_size = (width if width > height else height)

    kernel_size_min = 0
    kernel_size_max = 0
    if blur_type == 1:
        kernel_size_min = int(kernel_bounds_mb[0] * img_size)
        kernel_size_max = int(kernel_bounds_mb[1] * img_size)
    elif blur_type == 2:
        kernel_size_min = int(kernel_bounds_fb[0] * img_size)
        kernel_size_max = int(kernel_bounds_fb[1] * img_size)
    elif blur_type == 3:
        kernel_size_min = int(kernel_bounds_mfb[0] * img_size)
        kernel_size_max = int(kernel_bounds_mfb[1] * img_size)

    if kernel_size_min < 3:
        kernel_size_min = 3
    if kernel_size_max < 3:
        kernel_size_max = 3

    if kernel_size_max > kernel_size_min:
        kernel_size = np.random.randint(kernel_size_min, kernel_size_max + 1)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    else:
        kernel_size = kernel_size_min

    # print 'blur_type = ' + str(blur_type)
    # print 'kernel_size = ' + str(kernel_size)

    return kernel_size


def apply_horizontal_motion(img):
    kernel_size = gen_kernel_size(img)

    kernel = np.zeros((kernel_size, kernel_size))
    if bool_rand():
        kernel[(kernel_size - 1) / 2] = np.ones(kernel_size) #горизонталь
    else:
        kernel[:, (kernel_size - 1) / 2] = np.ones(kernel_size) #вертикаль

    kernel = kernel / kernel_size
    return cv2.filter2D(img, -1, kernel)


def apply_diagonal_motion(img):
    kernel_size = gen_kernel_size(img)

    kernel = np.eye(kernel_size) #главная диагональ
    if bool_rand():
        kernel = np.rot90(kernel) #побочная диагональ
    # else:
    #     pass #главная диагональ

    kernel = kernel / kernel_size
    return cv2.filter2D(img, -1, kernel)


def do_motion_blur(img):
    # print "do_motion_blur"
    do = np.random.randint(4)
    if do == 0:
        # print "horiz + diag"
        blured = apply_horizontal_motion(img) #размытие по горизонали и вертикали
        blured = apply_diagonal_motion(blured)  # размытие по диагонали
    elif do == 1:
        # print "diag + horiz"
        blured = apply_diagonal_motion(img)
        blured = apply_horizontal_motion(blured)
    elif do == 2:
        # print "horiz"
        blured = apply_horizontal_motion(img)
    elif do == 3:
        # print "diag"
        blured = apply_diagonal_motion(img)
    return blured


def do_focus_blur(img):
    # print "do_focus_blur"
    kernel_size = gen_kernel_size(img)
    blured = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return blured


def do_motion_focus_blur(img):
    """
    Порядок наложения размытия:
    - фокусное размытие
    - размытие движения
    """
    # print "do_motion_focus_blur"
    blured = do_focus_blur(img)
    blured = do_motion_blur(blured)
    return blured


'''
ОБРЕЗКА ФОТО ДО 500x375
'''
const_w = 500
const_h = 375

def crop_img(img):
    height, width = img.shape[:2]

    # поворот наибольшей стороной по горизонтали
    if height > width:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        height, width = img.shape[:2]

    if height >= const_h:
        img = img[:const_h, :const_w]
    else:
        img = img[:, :const_w]
        bottom_size = const_h - height
        img = cv2.copyMakeBorder(img, 0, bottom_size, 0, 0, cv2.BORDER_REFLECT_101)

    return img


'''
ИТОГОВАЯ ГЕНЕРАЦИЯ РАЗМЫТИЯ + ШУМА
'''

def blur_and_noise_images(path):

    files = glob.glob(path)
    for f in files:
        if os.path.isfile(f) and f.lower().endswith(".jpeg"):
            img = cv2.imread(f)
            height, width, channels = img.shape
            print f
            path_to_save, basename = os.path.split(f)
            # name, ext = os.path.splitext(os.path.basename(f))
            name, ext = os.path.splitext(basename)

            global blur_type
            blur_type = 1
            img_mb = do_motion_blur(img)
            blur_type = 2
            img_fb = do_focus_blur(img)
            blur_type = 3
            img_mfb = do_motion_focus_blur(img)

            img_mb_wn = add_noise(img_mb)
            img_fb_wn = add_noise(img_fb)
            img_mfb_wn = add_noise(img_mfb)

            crop_mb = crop_img(img_mb_wn)
            crop_fb = crop_img(img_fb_wn)
            crop_mfb = crop_img(img_mfb_wn)

            # cv2.imwrite(path_to_save + '/' + name + '_mb' + ext, img_mb)
            # cv2.imwrite(path_to_save + '/' + name + '_fb' + ext, img_fb)
            # cv2.imwrite(path_to_save + '/' + name + '_mfb' + ext, img_mfb)

            # cv2.imwrite(path_to_save + '/' + name + '_mb' + ext, img_mb_wn)
            # cv2.imwrite(path_to_save + '/' + name + '_fb' + ext, img_fb_wn)
            # cv2.imwrite(path_to_save + '/' + name + '_mfb' + ext, img_mfb_wn)

            cv2.imwrite(path_to_save + '/' + name + '_mb' + ext, crop_mb)
            cv2.imwrite(path_to_save + '/' + name + '_fb' + ext, crop_fb)
            cv2.imwrite(path_to_save + '/' + name + '_mfb' + ext, crop_mfb)

        elif os.path.isdir(f):
            blur_and_noise_images(f + "/*")


if __name__ == "__main__":
    blur_and_noise_images('/home/doleinik/SharpeningPhoto/quality_ImageNet/test_500/images/*')