#!/usr/bin/env sh

DATA=data
TOOLS=/home/caffe/build/tools

$TOOLS/compute_image_mean train_lmdb $DATA/train_image_mean.binaryproto
echo "Done train mean."

$TOOLS/compute_image_mean test_lmdb $DATA/test_image_mean.binaryproto
echo "Done test mean."

$TOOLS/compute_image_mean val_lmdb $DATA/val_image_mean.binaryproto
echo "Done val mean."