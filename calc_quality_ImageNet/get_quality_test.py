# -*- coding: utf-8 -*-
import numpy
import glob
import os
import cv2

ksize_laplacian = 3
ksize_median = 3

def calc_of_laplacian(gray_img, type_of_calc):
    gray_img = cv2.resize(gray_img, (512, 512), interpolation=cv2.INTER_CUBIC)
    median = cv2.medianBlur(gray_img, ksize_median)
    laplacian = cv2.Laplacian(median, -1, ksize=ksize_laplacian)
    if type_of_calc == 'std':
        return laplacian.std()
    elif type_of_calc == 'mean':
        return laplacian.mean()


def get_image_paths(path, good_img_paths, bad_img_paths, img_std, img_mean):

    files = glob.glob(path)
    for f in files:
        if os.path.isfile(f) and f.lower().endswith(".jpeg"):
            print f

            img = cv2.imread(f, 0)
            height, width = img.shape

            if height >= 512 or width >= 512:
                std = calc_of_laplacian(img, 'std')
                mean = calc_of_laplacian(img, 'mean')

                # if std > 26.0 and mean > 10.0:
                good_img_paths.append(f)
                img_std.append(std)
                img_mean.append(mean)
                # else:
                #     bad_img_paths.append(f)
            else:
                # pass
                bad_img_paths.append(f)

        elif os.path.isdir(f):
            get_image_paths(f + "/*", good_img_paths, bad_img_paths, img_std, img_mean)


if __name__ == "__main__":

    good_img_paths = [] #фото хорошего качества
    bad_img_paths = [] #фото плохого качества
    img_std = []
    img_mean = []

    path = '/home/image-net/ILSVRC2015/Data/CLS-LOC/test/*'
    get_image_paths(path, good_img_paths, bad_img_paths, img_std, img_mean)

    save_path = 'test/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    numpy.save(save_path+'good_img_paths.npy', numpy.array(good_img_paths))
    numpy.save(save_path+'bad_img_paths.npy', numpy.array(bad_img_paths))
    numpy.save(save_path+'img_std.npy', numpy.array(img_std))
    numpy.save(save_path+'img_mean.npy', numpy.array(img_mean))
