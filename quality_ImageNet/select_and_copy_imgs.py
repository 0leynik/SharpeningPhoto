# -*- coding: utf-8 -*-
import numpy
import matplotlib.pyplot as plt
import os
import shutil

if __name__ == "__main__":

    size_img = 500
    do_copy = True

    selected_img_paths = [[], [], []]
    type_hist = ['train_'+str(size_img),'val_'+str(size_img),'test_'+str(size_img)]

    for i in range(3):
        path = type_hist[i] + '/'
        good_img_paths = numpy.load(path+'good_img_paths.npy')
        bad_img_paths = numpy.load(path+'bad_img_paths.npy')
        img_std = numpy.load(path+'img_std.npy')
        img_mean = numpy.load(path+'img_mean.npy')

        print('*** Набор ' + type_hist[i] + ' ***')
        print('Все файлы : ' + str(len(good_img_paths)))

        path_to_save = type_hist[i] + '/images/'
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        threshold_std = img_std.mean() + img_std.std()
        threshold_mean = img_mean.mean() + img_mean.std()

        for n_img in range(len(good_img_paths)):
            if img_std[n_img] >= threshold_std and img_mean[n_img] >= threshold_mean:
                selected_img_paths[i].append(good_img_paths[n_img])
                if do_copy:
                    print(good_img_paths[n_img] + ' -> ' + str(len(selected_img_paths[i])))
                    shutil.copy(good_img_paths[n_img], path_to_save + str(len(selected_img_paths[i])) + '.JPEG')

        print('Отобранные резкие : ' + str(len(selected_img_paths[i])) + '\n')

        print('img_std mean = ' + str(img_std.mean()))
        print('img_std std = ' + str(img_std.std()))
        print('\nimg_std mean + std = ' + str(threshold_std) + '\n')

        print('img_mean mean = ' + str(img_mean.mean()))
        print('img_mean std = ' + str(img_mean.std()))
        print('\nimg_mean mean + std = ' + str(threshold_mean) + '\n\n')

        # plt.figure(type_hist[i])
        # plt.hist(img_std, bins = 500, histtype = 'stepfilled', color='b', alpha = 0.5)  # plt.hist passes it's arguments to np.histogram
        # plt.hist(img_mean, bins = 500, histtype = 'bar', color='r', alpha = 0.5)  # plt.hist passes it's arguments to np.histogram
        # # plt.xlabel('Value STD or MEAN')
        # # plt.ylabel('Count Images')
        # plt.legend(['STD','MEAN'])
        # # plt.title(type_hist[i].upper() + ' (Histogram of STD and MEAN)')
        # # plt.title(type_hist[i].upper())
        # plt.grid(True, linestyle='--')
        # plt.xticks(range(0,101,10))
    # plt.show()