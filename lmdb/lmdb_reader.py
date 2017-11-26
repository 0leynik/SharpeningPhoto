# -*- coding: utf-8 -*-
import caffe
import lmdb
import numpy as np
import matplotlib.pyplot as plt
import cv2

# https://chrischoy.github.io/research/reading-protobuf-db-in-python/
# http://deepdish.io/2015/04/28/creating-lmdb-in-python/
# https://gist.github.com/bearpaw/3a07f0e8904ed42f376e
# http://research.beenfrog.com/code/2015/12/30/write-read-lmdb-example.html
# http://shengshuyang.github.io/hook-up-lmdb-with-caffe-in-python.html

visualize = True
lmdb_path = "train_blur_lmdb"

env = lmdb.open(lmdb_path, readonly=True)
with env.begin() as txn:
    with txn.cursor() as curs:
        datum = caffe.proto.caffe_pb2.Datum()

        for key, value in curs:

            datum.ParseFromString(value)

            label = datum.label
            data = caffe.io.datum_to_array(datum)
            # (datum.channels, datum.height, datum.width)

            # if label == 999:
            print "key: ", key
            print "key type: ", type(key)
            # print "value ", value
            print "value type: ", type(value)

            print "datum.label ", label
            # print "datum.data ", data
            print "type(datum.data) ", type(data)
            print "datum.data ", data.dtype
            print "datum.data.shape ", data.shape

            if visualize:
                # CxHxW -> HxWxC
                img = np.transpose(data, (1, 2, 0))
                print "img.shape ", img.shape
                # BGR -> RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # matplotlib.pyplot.imshow()
                # HxWx3 – RGB (float or uint8 array)
                plt.imshow(img)
                plt.show()