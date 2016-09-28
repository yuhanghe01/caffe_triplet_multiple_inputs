#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.

ROOT_DIR=/home/heyuhang/caffe-sl-triplet
EXAMPLE=$ROOT_DIR/examples/cifar10
DATA=$ROOT_DIR/data/cifar10
DBTYPE=lmdb

echo "Creating $DBTYPE..."

rm -rf $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/cifar10_test_$DBTYPE

$ROOT_DIR/build/examples/cifar10/convert_cifar_data.bin $DATA $EXAMPLE $DBTYPE

echo "Computing image mean..."

$ROOT_DIR/build/tools/compute_image_mean -backend=$DBTYPE \
  $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/mean.binaryproto

echo "Done."
