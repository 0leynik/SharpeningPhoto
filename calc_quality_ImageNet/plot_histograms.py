# -*- coding: utf-8 -*-
import numpy
import matplotlib.pyplot as plt
import os
import shutil

if __name__ == "__main__":

    filtered_img_paths = [[],[],[]]
    type_hist = ['train','val','test']

    for i in range(3):
        path = type_hist[i] + '/'
        good_img_paths = numpy.load(path+'good_img_paths.npy')
        bad_img_paths = numpy.load(path+'bad_img_paths.npy')
        img_std = numpy.load(path+'img_std.npy')
        img_mean = numpy.load(path+'img_mean.npy')

        # mean>20
        # std>50
        print(type_hist[i])
        print(len(good_img_paths))
        path_to_save = type_hist[i] + '/images/'
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        for n_img in range(len(good_img_paths)):
            if img_std[n_img] >= 46.0 and img_mean[n_img] >= 22.0:
                filtered_img_paths[i].append(good_img_paths[n_img])
                # print(good_img_paths[n_img] + ' -> ' + str(len(filtered_img_paths[i])))
                # shutil.copy(good_img_paths[n_img], path_to_save + str(len(filtered_img_paths[i])) + '.JPEG')

        print(len(filtered_img_paths[i]))
        print

        print('среднее значение img_std ='+str(img_std.mean()))
        print('сигма 1 ='+str(img_std.std()))
        print('+сигма 1 ='+str(img_std.mean()+img_std.std()))

        print('среднее значение img_mean =' + str(img_mean.mean()))
        print('сигма 1 =' + str(img_mean.std()))
        print('+сигма 1 =' + str(img_mean.mean() + img_mean.std()))
        print

        plt.figure(type_hist[i])
        plt.hist(img_std, bins = 500, histtype = 'stepfilled', color='b', alpha = 0.5)  # plt.hist passes it's arguments to np.histogram
        plt.hist(img_mean, bins = 500, histtype = 'bar', color='r', alpha = 0.5)  # plt.hist passes it's arguments to np.histogram
        # plt.xlabel('Value STD or MEAN')
        # plt.ylabel('Count Images')
        plt.legend(['STD','MEAN'])
        # plt.title(type_hist[i].upper() + ' (Histogram of STD and MEAN)')
        # plt.title(type_hist[i].upper())
        plt.grid(True, linestyle='--')
        plt.xticks(range(0,101,10))
    plt.show()
'''

train
1058245
130452

среднее значение img_std =33.0686494933
сигма 1 =12.5123470801
+сигма 1 =45.5809965734
среднее значение img_mean =14.3513094243
сигма 1 =7.65445840705
+сигма 1 =22.0057678313

val
46769
6010

среднее значение img_std =33.4619545369
сигма 1 =12.5461227112
+сигма 1 =46.0080772482
среднее значение img_mean =14.4814246731
сигма 1 =7.66325526641
+сигма 1 =22.1446799395

test
93578
11964

среднее значение img_std =33.5127538381
сигма 1 =12.4659421553
+сигма 1 =45.9786959934
среднее значение img_mean =14.5001429629
сигма 1 =7.61068048254
+сигма 1 =22.1108234454

'''