# -*- coding: utf-8 -*-
import os
import glob
import cv2
import numpy
from matplotlib import pyplot as plt

ksize_laplacian = 5
ksize_median = 9
# ksize_median = 9+8

def get_image_paths(path, image_paths):
    """
    получить пути изображений
    """

    files = glob.glob(path)

    for f in files:

        if os.path.isfile(f):
            if f.lower().endswith("_mb.jpg"):
                pass # print # image_paths[1].append(f)
            elif f.lower().endswith("_fb.jpg"):
                pass # print # image_paths[2].append(f)
            elif f.lower().endswith("_mfb.jpg"):
                pass # print # image_paths[3].append(f)
            elif f.lower().endswith(".jpg"):
                image_paths.append(
                    [
                        f,
                        f[:-4] + "_mb.JPG",
                        f[:-4] + "_fb.JPG",
                        f[:-4] + "_mfb.JPG"
                    ]
                )
        elif os.path.isdir(f):
            get_image_paths(f + "/*", image_paths)
def print_image_paths(image_paths):
    """
    вывести пути изображений
    """
    print("\n----------> IMAGES <----------\n")
    for imgs in image_paths:
        for i in range(4):
            print(imgs[i])
        print


def calc_handmade_photo(type_of_calc, image_paths, print_calc = False):
    """
    Оценка размытия фото сделанных вручную
    """
    print('Вычисление '+type_of_calc+' для каждого изображения')
    # вычисление sdt или mean для каждого изображения
    evaluation_values = [] #значения оценки
    for image_path in image_paths:
        values = []
        for i in range(4):
            gray = cv2.imread(image_path[i], 0)
            median = cv2.medianBlur(gray,  ksize_median)
            laplacian = cv2.Laplacian(median, -1, ksize=ksize_laplacian)
            if type_of_calc == 'std':
                values.append(laplacian.std())
            elif type_of_calc == 'mean':
                values.append(laplacian.mean())
        evaluation_values.append(values)

    # вычисление разницы (отклонения) между резким и размытым по каждому изображению
    diff_values = [] #разница значений
    for value in evaluation_values:
        diff = []
        for i in range(4):
            diff.append(value[i] / value[0])
        diff_values.append(diff)

    # расчет среднего значения по всем типам размытия
    mean_values = numpy.array(diff_values).mean(axis=0)
    mean_values = mean_values.tolist()

    if print_calc:
        print("\n----------> Calc of " + type_of_calc + " <----------")
        print("\nDiff of " + type_of_calc + ":")
        for i in range(len(image_paths)):
            print(image_paths[i][0])
            for j in range(4):
                print("diff = " + str(diff_values[i][j]))
            print("")

        print("\nMean of diff " + type_of_calc + ":")
        for value in mean_values:
            print(value)

    return diff_values


def bool_random():
    return numpy.random.choice([True, False])
def bgr_to_rgb(bgr_img):
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

kernel_bounds = [15.0, 71.0]
# 15*500/4320=1,736111111111111
# 71*500/4320=8,217592592592593

def gen_mb_kernel_size(image): # определение размера ядра свертки
    height, width, channels = image.shape
    kernel_size = 0
    img_size = (width if width > height else height)
    kernel_size_min = int((kernel_bounds[0] / 4320)*img_size)
    kernel_size_max = int((kernel_bounds[1] / 4320)*img_size)

    if kernel_size_min < 3:
        kernel_size_min = 3
    if kernel_size_max < 3:
        kernel_size_max = 3

    # print (kernel_size_min, kernel_size_max)
    if kernel_size_max > kernel_size_min:
        kernel_size = numpy.random.randint(kernel_size_min, kernel_size_max + 1)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    else:
        kernel_size = kernel_size_min

    return kernel_size
def gen_fb_kernel_size(image): # определение размера ядра свертки
    height, width, channels = image.shape
    kernel_size = 0
    img_size = (width if width > height else height)
    kernel_size_min = int((kernel_bounds[0] / 4320) * img_size)
    kernel_size_max = int((kernel_bounds[1] / 4320) * img_size)

    if kernel_size_min < 3:
        kernel_size_min = 3
    if kernel_size_max < 3:
        kernel_size_max = 3

    # print (kernel_size_min, kernel_size_max)
    if kernel_size_max > kernel_size_min:
        kernel_size = numpy.random.randint(kernel_size_min, kernel_size_max + 1)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    else:
        kernel_size = kernel_size_min

    return kernel_size

def apply_horizontal_motion(image, kernel_size = 0):
    if kernel_size == 0:
        kernel_size = gen_mb_kernel_size(image)
    # print("horiz kernel size =", kernel_size)
    kernel = numpy.zeros((kernel_size, kernel_size))
    '''!!!!!!!!!!!!!!!!ВЕРНУТЬ ОБРАТНО!!!!!!!!!!!!!!!!!!!!!'''
    kernel[(kernel_size - 1) / 2] = numpy.ones(kernel_size)  # горизонталь

    # if bool_random():
    #    kernel[(kernel_size - 1) / 2] = numpy.ones(kernel_size) #горизонталь
    # else:
    #     kernel[:, (kernel_size - 1) / 2] = numpy.ones(kernel_size) #вертикаль

    kernel = kernel / kernel_size
    # applying the kernel to the image
    return cv2.filter2D(image, -1, kernel)
def apply_diagonal_motion(image, kernel_size = 0):
    if kernel_size == 0:
        kernel_size = gen_mb_kernel_size(image)
    # print("diag kernel size =",kernel_size)
    kernel = numpy.eye(kernel_size)

    '''!!!!!!!!!!!!!!!!ВЕРНУТЬ ОБРАТНО!!!!!!!!!!!!!!!!!!!!!'''
    # if bool_random():
    #     kernel = numpy.rot90(kernel) #побочная диагональ

    # else:
    #     pass #главная диагональ
    kernel = kernel / kernel_size
    # applying the kernel to the image
    return cv2.filter2D(image, -1, kernel)


def do_motion_blur(image_paths=None, images=None, kernel_size=0, plot=False):
    if image_paths is not None:
        images = []
        for image_path in image_paths:
            images.append(cv2.imread(image_path[0]))


    motion_blur_images = []
    for original in images:

        blured = original

        do = numpy.random.randint(4)

        do = 3

        if do == 0:
            # print "horiz + diag"
            blured = apply_horizontal_motion(blured, kernel_size) #размытие по горизонали и вертикали
            blured = apply_diagonal_motion(blured, kernel_size)  # размытие по диагонали
        elif do == 1:
            # print "diag + horiz"
            blured = apply_diagonal_motion(blured, kernel_size)
            blured = apply_horizontal_motion(blured, kernel_size)
        elif do == 2:
            # print "horiz"
            blured = apply_horizontal_motion(blured, kernel_size)
        elif do == 3:
            # print "diag"
            blured = apply_diagonal_motion(blured, kernel_size)

        motion_blur_images.append(blured)

    if plot == True and image_paths is not None:
        for i in range(len(image_paths)):
            # plt.subplot(231), plt.imshow(bgr_to_rgb(cv2.imread(image_path[0]))), plt.title('Original'), plt.axis('off')
            # plt.subplot(232), plt.imshow(bgr_to_rgb(motion_blur_images[i])), plt.title('Motion Blur'), plt.axis('off')
            # plt.subplot(233), plt.imshow(bgr_to_rgb(cv2.imread(image_path[1]))), plt.title('Motion Blur Foto'), plt.axis('off')
            #
            # # plt.xlim(0, 1000), plt.ylim(1000, 0)
            # plt.subplot(234), plt.imshow(bgr_to_rgb(cv2.imread(image_path[0]))), plt.title('Original'), plt.axis([0, 1000, 1000, 0]), plt.axis('off')
            # plt.subplot(235), plt.imshow(bgr_to_rgb(motion_blur_images[i])), plt.title('Motion Blur'), plt.axis([0, 1000, 1000, 0]), plt.axis('off')
            # plt.subplot(236), plt.imshow(bgr_to_rgb(cv2.imread(image_path[1]))), plt.title('Motion Blur Foto'), plt.axis([0, 1000, 1000, 0]), plt.axis('off')

            plt.subplot(331), plt.imshow(bgr_to_rgb(cv2.imread(image_paths[i][0]))), plt.title('Original'), plt.axis('off')
            plt.subplot(332), plt.imshow(bgr_to_rgb(motion_blur_images[i])), plt.title('Motion Blur'), plt.axis('off')
            plt.subplot(333), plt.imshow(bgr_to_rgb(cv2.imread(image_paths[i][1]))), plt.title('Motion Blur Foto'), plt.axis('off')

            plt.subplot(334), plt.imshow(bgr_to_rgb(cv2.imread(image_paths[i][0]))), plt.title('Original'), plt.axis([0, 1000, 1000, 0]), plt.axis('off')
            plt.subplot(335), plt.imshow(bgr_to_rgb(motion_blur_images[i])), plt.title('Motion Blur'), plt.axis([0, 1000, 1000, 0]), plt.axis('off')
            plt.subplot(336), plt.imshow(bgr_to_rgb(cv2.imread(image_paths[i][1]))), plt.title('Motion Blur Foto'), plt.axis([0, 1000, 1000, 0]), plt.axis('off')

            plt.subplot(337), plt.imshow(bgr_to_rgb(cv2.imread(image_paths[i][0]))), plt.title('Original'), plt.axis([1000, 2000, 2000, 1000]), plt.axis('off')
            plt.subplot(338), plt.imshow(bgr_to_rgb(motion_blur_images[i])), plt.title('Motion Blur'), plt.axis([1000, 2000, 2000, 1000]), plt.axis('off')
            plt.subplot(339), plt.imshow(bgr_to_rgb(cv2.imread(image_paths[i][1]))), plt.title('Motion Blur Foto'), plt.axis([1000, 2000, 2000, 1000]), plt.axis('off')

            plt.show()

            # cv2.imshow('No Motion Blur', original)
            # cv2.imshow('Motion Blur', blured)
            # cv2.imshow('Motion Blur Foto', cv2.imread(image_path[1]))
            # cv2.waitKey()

    return motion_blur_images
def do_focus_blur(image_paths=None, images=None, kernel_size=0, plot=False):

    if image_paths is not None:
        images = []
        for image_path in image_paths:
            images.append(cv2.imread(image_path[0]))

    focus_blur_images = []
    for original in images:

        if kernel_size == 0:
            kernel_size = gen_fb_kernel_size(original)

        blured = cv2.GaussianBlur(original, (kernel_size, kernel_size), 0)
        # blured = cv2.blur(original,(kernel_size, kernel_size))

        focus_blur_images.append(blured)


    if plot == True and image_paths is not None:
        for i in range(len(image_paths)):
            plt.subplot(331), plt.imshow(bgr_to_rgb(cv2.imread(image_paths[i][0]))), plt.title('Original'), plt.axis('off')
            plt.subplot(332), plt.imshow(bgr_to_rgb(focus_blur_images[i])), plt.title('Gaussian Blur'), plt.axis('off')
            plt.subplot(333), plt.imshow(bgr_to_rgb(cv2.imread(image_paths[i][2]))), plt.title('Focus Blur Foto'), plt.axis('off')

            plt.subplot(334), plt.imshow(bgr_to_rgb(cv2.imread(image_paths[i][0]))), plt.title('Original'), plt.axis([0, 1000, 1000, 0]), plt.axis('off')
            plt.subplot(335), plt.imshow(bgr_to_rgb(focus_blur_images[i])), plt.title('Gaussian Blur'), plt.axis([0, 1000, 1000, 0]), plt.axis('off')
            plt.subplot(336), plt.imshow(bgr_to_rgb(cv2.imread(image_paths[i][2]))), plt.title('Focus Blur Foto'), plt.axis([0, 1000, 1000, 0]), plt.axis('off')

            plt.subplot(337), plt.imshow(bgr_to_rgb(cv2.imread(image_paths[i][0]))), plt.title('Original'), plt.axis([1000, 2000, 2000, 1000]), plt.axis('off')
            plt.subplot(338), plt.imshow(bgr_to_rgb(focus_blur_images[i])), plt.title('Gaussian Blur'), plt.axis([1000, 2000, 2000, 1000]), plt.axis('off')
            plt.subplot(339), plt.imshow(bgr_to_rgb(cv2.imread(image_paths[i][2]))), plt.title('Focus Blur Foto'), plt.axis([1000, 2000, 2000, 1000]), plt.axis('off')

            plt.show()

    return focus_blur_images
def do_motion_focus_blur(image_paths=None, images=None, kernel_size=0, plot=False):
    """
    Порядок наложения фильтров:
    - фокусное размытие
    - размытие движения
    """
    if image_paths is not None:
        images = []
        for image_path in image_paths:
            images.append(cv2.imread(image_path[0]))

    motion_focus_blur_images = do_focus_blur(None, images, kernel_size, plot)
    motion_focus_blur_images = do_motion_blur(None, motion_focus_blur_images, kernel_size, plot)

    if plot == True and image_paths is not None:
        for i in range(len(image_paths)):
            plt.subplot(331), plt.imshow(bgr_to_rgb(cv2.imread(image_paths[i][0]))), plt.title('Original'), plt.axis('off')
            plt.subplot(332), plt.imshow(bgr_to_rgb(motion_focus_blur_images[i])), plt.title('MF Blur'), plt.axis('off')
            plt.subplot(333), plt.imshow(bgr_to_rgb(cv2.imread(image_paths[i][3]))), plt.title('MF Blur Foto'), plt.axis('off')

            plt.subplot(334), plt.imshow(bgr_to_rgb(cv2.imread(image_paths[i][0]))), plt.title('Original'), plt.axis([0, 1000, 1000, 0]), plt.axis('off')
            plt.subplot(335), plt.imshow(bgr_to_rgb(motion_focus_blur_images[i])), plt.title('MF Blur'), plt.axis([0, 1000, 1000, 0]), plt.axis('off')
            plt.subplot(336), plt.imshow(bgr_to_rgb(cv2.imread(image_paths[i][3]))), plt.title('MF Blur Foto'), plt.axis([0, 1000, 1000, 0]), plt.axis('off')

            plt.subplot(337), plt.imshow(bgr_to_rgb(cv2.imread(image_paths[i][0]))), plt.title('Original'), plt.axis([1000, 2000, 2000, 1000]), plt.axis('off')
            plt.subplot(338), plt.imshow(bgr_to_rgb(motion_focus_blur_images[i])), plt.title('MF Blur'), plt.axis([1000, 2000, 2000, 1000]), plt.axis('off')
            plt.subplot(339), plt.imshow(bgr_to_rgb(cv2.imread(image_paths[i][3]))), plt.title('MF Blur Foto'), plt.axis([1000, 2000, 2000, 1000]), plt.axis('off')

            plt.show()

    return motion_focus_blur_images


def calc_kernel_size(image_paths, original_diff_values):
    """
    Расчет диапазона размеров ядер свертки для каждого типа размытия
    """

    # папка для сохранения изображений
    # размытых до степени реального размытия
    dir_to_save = 'calc_kernel_size(median=' + str(ksize_median) + ')'
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)

    # расчет значений std для всех резких изображений
    std_originals = []
    for image_path in image_paths:
        gray = cv2.imread(image_path[0], 0)
        median = cv2.medianBlur(gray, ksize_median)
        calc_laplacian = cv2.Laplacian(median, -1, ksize=ksize_laplacian)
        std_originals.append(calc_laplacian.std())

        # inverts = cv2.bitwise_not(calc_laplacian)
        #
        # cv2.imshow('orig', inverts)
        # # cv2.imshow('orig', cv2.imread(image_path[0]))
        #
        # gray = cv2.imread(image_path[2], 0)
        # median = cv2.medianBlur(gray, ksize_median)
        # calc_laplacian = cv2.Laplacian(median, -1, ksize=ksize_laplacian)
        # inverts = cv2.bitwise_not(calc_laplacian)
        #
        # cv2.imshow('orig_mb', inverts)
        # # cv2.imshow('orig_mb', cv2.imread(image_path[1]))




    type_calc = ["Вычисляем размытие движения!", "Вычисляем фокусное размытие!", "Вычисляем фокусное размытие + размытие движения!"]
    type_kernel = ['mb', 'fb', 'mfb']

    # 0 motion
    # 1 focus
    # 2 motion-focus
    for num_blur in range(3):
        print('\n\n' + type_calc[num_blur])
        kernel = 3
        iterations = [0] * len(image_paths)

        while True:
            print('\nkernel = ' + str(kernel))

            blured_images = []
            if num_blur == 0:
                blured_images = do_motion_blur(image_paths, None, kernel, False)
            elif num_blur == 1:
                blured_images = do_focus_blur(image_paths, None, kernel, False)
            elif num_blur == 2:
                blured_images = do_motion_focus_blur(image_paths, None, kernel, False)


            std_blured_images = [] # вычисление Laplacian.std() для каждого изображения
            diff_values = [] # вычисление разницы (отклонения) между резким и размытым по каждому изображению

            for i in range(len(image_paths)):
                if iterations[i] != 0:
                    std_blured_images.append(0.)
                    diff_values.append(0.)
                    continue

                gray = cv2.cvtColor(blured_images[i], cv2.COLOR_BGR2GRAY)
                median = cv2.medianBlur(gray, ksize_median)
                calc_laplacian = cv2.Laplacian(median, -1, ksize=ksize_laplacian)
                # inverts = cv2.bitwise_not(calc_laplacian)

                std_blured_images.append(calc_laplacian.std())
                diff_values.append(std_blured_images[i] / std_originals[i])

                print(image_paths[i][0] + ' kernel=' + str(kernel) + ' diff(' + str(original_diff_values[i][num_blur+1])+':'+str(diff_values[i])+')')


                if diff_values[i] < original_diff_values[i][num_blur+1] and iterations[i] == 0:
                    # cv2.imshow(str(diff_values[i]), inverts)
                    # cv2.waitKey()
                    iterations[i] = [image_paths[i][0], kernel, original_diff_values[i][num_blur+1], diff_values[i]]
                    path_to_save, basename = os.path.split(image_paths[i][0])
                    name, ext = os.path.splitext(basename)

                    # сохранение размытой картинки
                    cv2.imwrite(dir_to_save+'/' + name + '_' + type_kernel[num_blur] + ' k=' + str(kernel) + ext, blured_images[i])


                    # создание резкой картинки без шума и добавление шума к разымтой
                    img = cv2.imread(image_paths[i][0])

                    print("img shape = " + str(img.shape))
                    print("img shape = " + str(type(img)))


                    # изображения без шумов
                    img_median3 = cv2.medianBlur(img, 3)
                    img_median5 = cv2.medianBlur(img, 5)
                    img_median7 = cv2.medianBlur(img, 7)
                    cv2.imwrite(dir_to_save+'/' + name + '_' + type_kernel[num_blur] + ' k=' + str(kernel) + ' m3' + ext, img_median3)
                    cv2.imwrite(dir_to_save+'/' + name + '_' + type_kernel[num_blur] + ' k=' + str(kernel) + ' m5' + ext, img_median5)
                    cv2.imwrite(dir_to_save+'/' + name + '_' + type_kernel[num_blur] + ' k=' + str(kernel) + ' m7' + ext, img_median7)

                    # чистый шум
                    noise3 = img - img_median3
                    noise5 = img - img_median5
                    noise7 = img - img_median7

                    print("noise mean = " + str(noise3.mean()))
                    print("noise mean = " + str(noise3.shape))
                    print("noise mean = " + str(type(noise3)))

                    # print("noise mean = " + str(noise3))
                    cv2.imwrite(dir_to_save+'/' + name + '_' + type_kernel[num_blur] + ' k=' + str(kernel) + ' n3' + ext, noise3)
                    cv2.imwrite(dir_to_save+'/' + name + '_' + type_kernel[num_blur] + ' k=' + str(kernel) + ' n5' + ext, noise5)
                    cv2.imwrite(dir_to_save+'/' + name + '_' + type_kernel[num_blur] + ' k=' + str(kernel) + ' n7' + ext, noise7)

                    # шум с порогом среднего значения по всем каналам
                    retval3, noise3_w_threshold = cv2.threshold(noise3, noise3.mean(), 255., cv2.THRESH_TRUNC)
                    retval5, noise5_w_threshold = cv2.threshold(noise5, noise5.mean(), 255., cv2.THRESH_TRUNC)
                    retval7, noise7_w_threshold = cv2.threshold(noise7, noise7.mean(), 255., cv2.THRESH_TRUNC)

                    # noise3_w_threshold = numpy.random.rand(3240, 4320, 3)*128
                    # noise5_w_threshold = numpy.random.rand(3240, 4320, 3)*128
                    # noise7_w_threshold = numpy.random.rand(3240, 4320, 3)*128

                    print("noise + th mean = " + str(noise3_w_threshold.mean()))
                    print("noise + th mean = " + str(noise3_w_threshold.shape))
                    print("noise + th mean = " + str(type(noise3_w_threshold)))
                    # print("noise mean = " + str(noise3_w_threshold))
                    cv2.imwrite(dir_to_save+'/' + name + '_' + type_kernel[num_blur] + ' k=' + str(kernel) + ' n3+th' + ext, cv2.bitwise_not(noise3_w_threshold))
                    cv2.imwrite(dir_to_save+'/' + name + '_' + type_kernel[num_blur] + ' k=' + str(kernel) + ' n5+th' + ext, cv2.bitwise_not(noise5_w_threshold))
                    cv2.imwrite(dir_to_save+'/' + name + '_' + type_kernel[num_blur] + ' k=' + str(kernel) + ' n7+th' + ext, noise7_w_threshold)


                    print('blured img type '+str(type(blured_images[i])))
                    print('blured img type '+str(blured_images[i].shape))


                    # добавление шума к размытым фото
                    blured_add_noise3 = blured_images[i] + noise3
                    blured_add_noise5 = blured_images[i] + noise5
                    blured_add_noise7 = blured_images[i] + noise7
                    cv2.imwrite(dir_to_save+'/' + name + '_' + type_kernel[num_blur] + ' k=' + str(kernel) + ' blur+n3' + ext, blured_add_noise3)
                    cv2.imwrite(dir_to_save+'/' + name + '_' + type_kernel[num_blur] + ' k=' + str(kernel) + ' blur+n5' + ext, blured_add_noise5)
                    cv2.imwrite(dir_to_save+'/' + name + '_' + type_kernel[num_blur] + ' k=' + str(kernel) + ' blur+n7' + ext, blured_add_noise7)

                    blured_add_noise3 = blured_images[i] + noise3_w_threshold
                    blured_add_noise5 = blured_images[i] + noise5_w_threshold
                    blured_add_noise7 = blured_images[i] + noise7_w_threshold
                    cv2.imwrite(dir_to_save+'/' + name + '_' + type_kernel[num_blur] + ' k=' + str(kernel) + ' blur+n3+th' + ext, blured_add_noise3)
                    cv2.imwrite(dir_to_save+'/' + name + '_' + type_kernel[num_blur] + ' k=' + str(kernel) + ' blur+n5+th' + ext, blured_add_noise5)
                    cv2.imwrite(dir_to_save+'/' + name + '_' + type_kernel[num_blur] + ' k=' + str(kernel) + ' blur+n7+th' + ext, blured_add_noise7)



            end_calc = 0 #выход когда все фото рассчитаны
            for iter in iterations:
                if iter == 0:
                    end_calc = 1

            if kernel >= 501 or end_calc == 0:
                break
            kernel += 8

        # вывод информаии об итерациях
        print('\nРезультат:')
        for iter in iterations:
            print iter


        iter_min = iterations[0][1]
        for iter in iterations:
            if iter_min > iter[1]:
                iter_min = iter[1]
        print('\nmin:' + str(iter_min))


        iter_max = iterations[0][1]
        for iter in iterations:
            if iter_max < iter[1]:
                iter_max = iter[1]
        print('\nmax:' + str(iter_max))

        # cropImages(dir_to_save)


def cropImages(crop_dir):
    img_paths = os.listdir(crop_dir)

    for i in range(len(img_paths)):
        img = cv2.imread(crop_dir+'/'+img_paths[i])
        cv2.imwrite(crop_dir+'/crop '+img_paths[i], img[950:2350, 1100:2600])
        # cv2.imwrite(crop_dir+'/crop '+img_paths[i], img[1150:2550, 0:1400])
        # [560:810, 900:1150]


def blur_images(path):

    files = glob.glob(path)
    for f in files:
        print f
        if os.path.isfile(f) and f.lower().endswith(".jpeg"):

                img = cv2.imread(f)
                height, width, channels = img.shape

                if height >= 500 or width >= 500:

                    path_to_save, basename = os.path.split(f)
                    # name, ext = os.path.splitext(os.path.basename(f))
                    name, ext = os.path.splitext(basename)

                    cv2.imwrite(path_to_save + '/' + name + '_mb'+ext, do_motion_blur(None,[img])[0])
                    cv2.imwrite(path_to_save + '/' + name + '_fb'+ext, do_focus_blur(None,[img])[0])
                    cv2.imwrite(path_to_save + '/' + name + '_mfb'+ext, do_motion_focus_blur(None,[img])[0])

                else:
                    pass

        elif os.path.isdir(f):
            blur_images(f + "/*")


if __name__ == "__main__":
    path = "/Users/dmitryoleynik/PycharmProjects/DeBlurringWithCNN/calc_kernel_sizes_for_bluring/original_foto/*"

    image_paths = []
    '''
    [ 0 - clear image.jpg,
      1 - motion blured image_mb.jpg,
      2 - focus blured image_fb.jpg,
      3 - motion-focus blured image_mfb.jpg
    ]
    '''

    get_image_paths(path, image_paths)
    print_image_paths(image_paths)


    diff_values = calc_handmade_photo('std', image_paths, True)
    calc_kernel_size(image_paths, diff_values)

    # cropImages('calc_kernel_size_9')
    # cropImages('calc_kernel_size_17')
    # cropImages('evaluation_foto_crop')
    # blur_images('/Users/dmitryoleynik/Desktop/test/*')

    print("\n\nCREATE FUNC TO BLUR IMAGE")

