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
lmdb_path = "/home/doleinik/l4/lmdb/train_lmdb"

env = lmdb.open(lmdb_path)
with env.begin() as txn:

    cursor = txn.cursor()
    datum = caffe.proto.caffe_pb2.Datum()

    for key, value in cursor:

        datum.ParseFromString(value)

        label = datum.label
        data = caffe.io.datum_to_array(datum)
        # (datum.channels, datum.height, datum.width)

        # if label == 999:
        print "key: ", key
        print "value ", value
        print "datum.label ", label
        print "datum.data ", data
        # print type(data)
        # print data.shape
        # print data.dtype


        if visualize:
            # CxHxW -> HxWxC
            img = np.transpose(data, (1, 2, 0))
            # BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # matplotlib.pyplot.imshow()
            # HxWx3 – RGB (float or uint8 array)
            plt.imshow(img)
            plt.show()