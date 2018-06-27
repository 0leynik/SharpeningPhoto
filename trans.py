# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import matplotlib as mpl
# mpl.use('Agg')
from scipy import interpolate

if __name__ == '__main__':

    img_shape = (200, 200)
    kernel_sizes = np.arange(1, 52, 2)

    for i in range(100):

        n = np.random.randint(2, 6)  # колечество точек
        if n == 2:
            k = 1  # степень полинома при интерполяции
        elif n == 3:
            k = 2
        else:
            k = 3

        # points = np.array([[50,0,0,50,50], np.fabs(np.array([100,50,0,50,0])-100)])+14
        # n=5
        # k=3

        points = np.random.randint(50, 151, (2, n))
        tck, u = interpolate.splprep(points, s=0, k=k)
        x, y = interpolate.splev(np.linspace(0, 1, 100), tck, der=0)

        spline_points = np.array([x, y], np.int32).T

        black = np.zeros(img_shape, np.uint8)

        thickness = np.random.randint(1, 41)
        cv2.polylines(black, [spline_points], False, 255, thickness)

        ksize = tuple(np.random.choice(kernel_sizes, 2))
        print(ksize)
        blured = cv2.GaussianBlur(black, ksize, 0)

        # cv2.imshow('n:' + str(n) + ' k:' + str(k) + ' thickness:' + str(thickness) + ' ksize:' + str(ksize), blured)

        indices = np.nonzero(blured)
        y_min, x_min = np.amin(indices, axis=1)
        y_max, x_max = np.amax(indices, axis=1)

        # print(x_min, x_max)
        # print(y_min, y_max)

        # print(blured[y_min:y_max, x_min:x_max].shape)

        y_max = y_max if (y_max-y_min) % 2 == 1 else (y_max + 1)
        x_max = x_max if (x_max-x_min) % 2 == 1 else (x_max + 1)

        final_kernel = blured[y_min:y_max, x_min:x_max]
        # print(final_kernel.shape)

        final_kernel_name = 'croped n:' + str(n) + ' k:' + str(k) + ' thickness:' + str(thickness) + ' ksize:' + str(ksize)
        # cv2.imshow(final_kernel_name, final_kernel )
        cv2.imwrite('kernels/'+final_kernel_name+'.png', final_kernel)

    cv2.waitKey(0)
    cv2.destroyAllWindows()