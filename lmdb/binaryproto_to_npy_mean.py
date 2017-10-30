import caffe
import numpy as np


# train_image_mean
BIN_MEAN_FILE = 'lmdb/data_lmdb/train_image_mean.binaryproto'
NPY_MEAN_FILE = 'lmdb/data_lmdb/train_image_mean.npy'

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( BIN_MEAN_FILE , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
np.save( NPY_MEAN_FILE , out )
print("Done train_image_mean.")


# val_image_mean
BIN_MEAN_FILE = 'lmdb/data_lmdb/val_image_mean.binaryproto'
NPY_MEAN_FILE = 'lmdb/data_lmdb/val_image_mean.npy'

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( BIN_MEAN_FILE , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
np.save( NPY_MEAN_FILE , out )
print("Done val_image_mean.")