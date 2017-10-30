#!/usr/bin/env sh
set -e

DATA=data
TOOLS=/home/caffe/build/tools

TRAIN_DATA_ROOT=data/train_imgs/
TEST_DATA_ROOT=data/test_imgs/
VAL_DATA_ROOT=data/val_imgs/

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  exit 1
fi
if [ ! -d "$TEST_DATA_ROOT" ]; then
  echo "Error: TEST_DATA_ROOT is not a path to a directory: $TEST_DATA_ROOT"
  exit 1
fi
if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  exit 1
fi


echo "Creating train lmdb..."
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    train_lmdb

echo "Creating test lmdb..."
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    $TEST_DATA_ROOT \
    $DATA/val.txt \
    test_lmdb

echo "Creating val lmdb..."
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    $VAL_DATA_ROOT \
    $DATA/val.txt \
    val_lmdb

echo "Done."