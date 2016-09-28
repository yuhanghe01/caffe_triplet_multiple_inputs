# Caffe Framework with various triplet loss neural network

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

This is my caffe framework which is initially designed for triplet loss neural network. It contains various useful implementations, including:

1. create LEVELDB/LMDB for triplet input: anchor/postive/negative. Code in ```tools/convert_triplet_db_dataset.cpp```.
2. create LEVELDB/LMDB for multiple postive and negative input. Code in ```tools/convert_multiple_triplet_db_dataset.cpp```.
3. loss layer with multiple postive and negative input. Code in ```src/caffe/layers/naive_triplet_multiple_loss_layer.cpp```.

If you find this useful, please cite Caffe paper:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
