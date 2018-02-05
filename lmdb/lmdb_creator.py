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
    for i in range(1, N + 1):
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

# const_w = 500
const_w = 128
# const_h = 375
const_h = 128
const_c = 3
batch_size = 1024

def create_lmdb(db_name, img_folder, img_names, N):

    # NumBytes = NumImages * shape[0] * shape[1] * shape[2] * sizeof(datatype) * sizeof(label)
    map_size = N * const_c * const_h * const_w * np.dtype(np.uint8).itemsize * np.dtype(np.int32).itemsize * 3
    # 901 307 250 000 bytes * 3 (коэффициент 3 на всякий случай)

    env = lmdb.open(db_name, map_size=map_size)
    print '\nLMDB \"' + db_name + '\" opened.\nStart writing!'

    # with env.begin(write=True) as txn:
    txn = env.begin(write=True)
    img_count = 0
    for i in range(N):
        img_count += 1
        print img_count
        # img = cv2.imread(img_folder + img_names[i])
        img = cv2.imread(img_folder + img_names[i])[:128, :128]
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

        if img_count % batch_size == 0:
            txn.commit()
            if img_count != N:
                txn = env.begin(write=True)
            print 'Commit img_count=' + str(img_count)

    if img_count % batch_size != 0:
        txn.commit()
        print 'Commit img_count=' + str(img_count)

    print 'Writing to \"' + db_name + '\" done!'

    env.close()
    print 'LMDB \"' + db_name + '\" closed.'


if __name__ == '__main__':

    print '*** train ***'
    N = 133527
    names = gen_shuffle_names(N)
    img_folder = '/home/doleinik/SharpeningPhoto/quality_ImageNet/train_500/images/'

    create_lmdb('train_blur_lmdb_128',
                img_folder,
                names[:, 0],
                N * 3)  # (mb+fb+mfb)
    create_lmdb('train_sharp_lmdb_128',
                img_folder,
                names[:, 1],
                N * 3)  # (mb+fb+mfb)

    print '*** test ***'
    N = 11853
    names = gen_shuffle_names(N)
    img_folder = '/home/doleinik/SharpeningPhoto/quality_ImageNet/test_500/images/'

    create_lmdb('test_blur_lmdb_128',
                img_folder,
                names[:, 0],
                N * 3)  # (mb+fb+mfb)
    create_lmdb('test_sharp_lmdb_128',
                img_folder,
                names[:, 1],
                N * 3)  # (mb+fb+mfb)

    print '*** val ***'
    N = 5936
    names = gen_shuffle_names(N)
    img_folder = '/home/doleinik/SharpeningPhoto/quality_ImageNet/val_500/images/'

    create_lmdb('val_blur_lmdb_128',
                img_folder,
                names[:, 0],
                N * 3)  # (mb+fb+mfb)
    create_lmdb('val_sharp_lmdb_128',
                img_folder,
                names[:, 1],
                N * 3)  # (mb+fb+mfb)

    print '\nComplete!'
