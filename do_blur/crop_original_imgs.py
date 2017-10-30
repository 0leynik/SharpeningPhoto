# -*- coding: utf-8 -*-
import os
import glob
import cv2
import numpy as np

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
обрезка изображений
'''
def crop_images(path, N):
    ext = '.JPEG'
    img_names = range(1, N+1)

    for img_name in img_names:
        str_name = str(img_name)
        print str_name

        img = cv2.imread(path + str_name + ext)
        croped = crop_img(img)
        cv2.imwrite(path + str_name + '_sh' + ext, croped)


if __name__ == "__main__":
    print 'train'
    crop_images('/home/doleinik/SharpeningPhoto/quality_ImageNet/train_500/images/', 133527)
    print 'train'
    crop_images('/home/doleinik/SharpeningPhoto/quality_ImageNet/test_500/images/', 11853)
    print 'train'
    crop_images('/home/doleinik/SharpeningPhoto/quality_ImageNet/val_500/images/', 5936)

    print 'Complete!'
