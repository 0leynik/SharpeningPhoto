# -*- coding: utf-8 -*-
import cv2
import numpy as np
import lmdb
import caffe


'''
ГЕНЕРАЦИЯ ПЕРЕМЕШАННЫХ ИМЕН ИЗОБРАЖЕНИЙ
'''

ext = '.JPEG'


def gen_shuffle_names(N):
    print '\nGen shuffle names:'
    img_names = []
    for i in range(1, N+1):
        i_str = str(i)

        sh = i_str + '_sh' + ext
        mb = i_str + '_mb' + ext
        fb = i_str + '_fb' + ext
        mfb = i_str + '_mfb' + ext

        img_names.append([mb, sh])
        img_names.append([fb, sh])
        img_names.append([mfb, sh])

    np_img_names = np.array(img_names)
    print np_img_names
    print np_img_names.shape

    np.random.shuffle(np_img_names)
    print np_img_names
    print np_img_names.shape

    # blur_img_names = np_img_names[:, 0]
    # sharp_img_names = np_img_names[:, 1]
    return np_img_names


'''
СОЗДАНИЕ LMDB
'''

const_w = 500
const_h = 375


def create_lmdb(db_name, img_folder, img_names, N):

    # NumBytes = NumImages * 3(mb+fb+mfb) * shape[0] * shape[1] * shape[2] * sizeof(datatype) * sizeof(label)
    map_size = N * 3 * 3 * const_h * const_w * np.dtype(np.uint8).itemsize * np.dtype(np.int32).itemsize * 10
    # 300 435 750 000 bytes * (коэффициент 3 на всякий случай)

    env = lmdb.open(db_name, map_size=map_size)
    print 'LMDB \"' + db_name + '\" opened.\nStart writing!'

    with env.begin(write=True) as txn:
        for i in range(N):
            print i
            img = cv2.imread(img_folder + img_names[i])
            # height, width, channels = img.shape

            # HxWxC -> CxHxW
            img = np.transpose(img, (2, 0, 1))
            channels, height, width = img.shape

            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = channels
            datum.height = height
            datum.width = width
            datum.data = img.tostring()
            # datum.data = img.tobytes()  # or .tostring() if numpy < 1.9
            datum.label = int(0)
            key_str = '{:08}'.format(i)

            txn.put(key_str.encode('ascii'), datum.SerializeToString())
    print 'Writing to \"' + db_name + '\" done!'

    env.close()
    print 'LMDB \"' + db_name + '\" closed.'


if __name__ == '__main__':

    print 'train'
    N = 133527
    names = gen_shuffle_names(N)
    img_folder = '/home/doleinik/SharpeningPhoto/quality_ImageNet/train_500/images/'

    create_lmdb('train_blur_lmdb',
                img_folder,
                names[:, 0],
                N)
    create_lmdb('train_sharp_lmdb',
                img_folder,
                names[:, 1],
                N)

    print 'test'
    N = 11853
    names = gen_shuffle_names(N)
    img_folder = '/home/doleinik/SharpeningPhoto/quality_ImageNet/test_500/images/'

    create_lmdb('test_blur_lmdb',
                img_folder,
                names[:, 0],
                N)
    create_lmdb('test_sharp_lmdb',
                img_folder,
                names[:, 1],
                N)

    print 'val'
    N = 5936
    names = gen_shuffle_names(N)
    img_folder = '/home/doleinik/SharpeningPhoto/quality_ImageNet/val_500/images/'

    create_lmdb('val_blur_lmdb',
                img_folder,
                names[:, 0],
                N)
    create_lmdb('val_sharp_lmdb',
                img_folder,
                names[:, 1],
                N)

    print '\nComplete!'
