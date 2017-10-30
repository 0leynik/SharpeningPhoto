# -*- coding: utf-8 -*-
import os
import glob
import cv2
import numpy as np

ext = '.JPEG'

def generate_list_lmdb(type_dataset, N):

    img_names = []
    for i in range(1, N+1):
        i_str = str(i)

        sh = i_str + '_sh' + ext
        mb = i_str + '_mb' + ext
        fb = i_str + '_fb' + ext
        mfb = i_str + '_mfb' + ext

        img_names.append( [mb, sh] )
        img_names.append( [fb, sh] )
        img_names.append( [mfb, sh] )

    np_img_names = np.array(img_names)
    print np_img_names
    print np_img_names.shape

    np.random.shuffle(np_img_names)
    print np_img_names
    print np_img_names.shape

    blur_img_names = np_img_names[:, 0]
    sharp_img_names = np_img_names[:, 1]

    np.savetxt('data/'+type_dataset+'_blur.txt', blur_img_names, fmt='%s')
    np.savetxt('data/'+type_dataset+'_sharp.txt', sharp_img_names, fmt='%s')
    print
    print blur_img_names
    print blur_img_names.shape
    print sharp_img_names
    print sharp_img_names.shape

if __name__ == '__main__':
    print 'train'
    generate_list_lmdb('train', 133527)
    print 'test'
    generate_list_lmdb('test', 11853)
    print 'val'
    generate_list_lmdb('val', 5936)

    print 'Complete!'
