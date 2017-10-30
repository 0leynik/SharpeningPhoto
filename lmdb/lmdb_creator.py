# -*- coding: utf-8 -*-
import cv2
import numpy as np
# import lmdb
# import caffe

N = 133527 * 3  # mb+fb+mfb
const_w = 500
const_h = 375

# def create_lmdb():
#
#     # NumBytes = NumImages * shape[0] * shape[1] * shape[2] * sizeof(datatype) * sizeof(label)
#     map_size = N * 3 * const_w * const_h * np.dtype(np.uint8).itemsize * np.dtype(np.int32).itemsize * 3
#     # 300 435 750 000 bytes * (коэффициент 3 на всякий случай)
#
#     env = lmdb.open('mylmdb', map_size=map_size)
#
#     with env.begin(write=True) as txn:
#         for i in range(N):
#
#             height, width, channels = img.shape
#
#             datum = caffe.proto.caffe_pb2.Datum()
#
#             datum.channels = channels
#             datum.height = height
#             datum.width = width
#             # datum.data = img.tobytes()  # or .tostring() if numpy < 1.9
#             datum.data = img.tostring()
#             datum.label = int(0)
#
#             keystr = '{:08}'.format(i)
#
#             txn.put(keystr.encode('ascii'), datum.SerializeToString())


if __name__ == '__main__':

    img = cv2.imread('8.JPEG');
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


    print img.shape

    cv2.imshow('img',img)
    cv2.waitKey()