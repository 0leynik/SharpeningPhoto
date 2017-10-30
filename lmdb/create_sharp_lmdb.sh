#!/usr/bin/env sh
set -e

DATA=/home/doleinik/SharpeningPhoto/lmdb/data
TOOLS=/home/caffe/build/tools

TRAIN_DATA_ROOT=/home/doleinik/SharpeningPhoto/quality_ImageNet/train_500/images/
TEST_DATA_ROOT=/home/doleinik/SharpeningPhoto/quality_ImageNet/test_500/images/
VAL_DATA_ROOT=/home/doleinik/SharpeningPhoto/quality_ImageNet/val_500/images/

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
    $DATA/train_sharp.txt \
    train_sharp_lmdb

echo "Creating test lmdb..."
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    $TEST_DATA_ROOT \
    $DATA/test_sharp.txt \
    test_sharp_lmdb

echo "Creating val lmdb..."
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    $VAL_DATA_ROOT \
    $DATA/val_sharp.txt \
    val_sharp_lmdb

echo "Done."