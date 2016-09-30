# Caffe Framework with various triplet loss neural network

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

This is my caffe framework which is initially designed for triplet loss neural network. It contains various useful implementations, including:

1. create LEVELDB/LMDB for triplet input: anchor/postive/negative. Code in ```tools/convert_triplet_db_dataset.cpp```.
2. create LEVELDB/LMDB for multiple postive and negative input. Code in ```tools/convert_multiple_triplet_db_dataset.cpp```.
3. loss layer with multiple postive and negative input. Code in ```src/caffe/layers/naive_triplet_multiple_loss_layer.cpp```.

## Create LMDB/LEVELDB file for triple input

To speed up the overall neural network training, it's better to convert input triple images into LMDB/LEVELDB. I provide the relevant code in `tools/convert_triplet_db_dataset.cpp`, in which each lines consists of anchor positive and negative respectively. An example list is shown below, not that all images should be shown with absolute pathes, separated by SPACE or TAB.

```!bash
anchor1.png      pos1.png      neg1.png
anchor2.png      pos2.png      neg2.png
...
```

## Create LMDB/LEVELDB file for multiple postives or negatives input

I extend the current triple input restraint to allow multiple postives and negatives input. That is, an anchor image corresponds to multiple postive and negative input. The relevant code is `tools/convert_multiple_triplet_db_dataset.cpp`, in which each line accordingly consists of anchor image, multiple postive and multiple negatie images, with two extra numbers indicating the the beginning of positive images and negatives images. An example is shown below:

```!bash
anchor1.png      pos1.png      pos2.png      pos3.png     neg1.png      neg2.png      neg3.png      neg4.png      1      4
...
```
## The loss layer of multiple positive and negative inputs.

I organize all the positive and negative inputs as one blob with the channel number equals to the total number of anchor positive and negative inputs. In the loss layer, I split the anchor positive and negative inputs and process them separately. So the loss layer prototxt should be specified with the length of anchor, positive and negative respectively.

```!bash
layer {
     name: "triplet_loss"
     type: "NaiveTripletMultipleLoss"
     bottom: "ip2norm"
     top: "loss"
     top: "accuracy"
     multiple_triplet_loss_param{
     margin: 1
     anchor_len: 1
     pos_len: 3
     neg_len: 4
     batch_size: 30
     sim_type: "DotProductSimilarity"
     }
}
```

If you find this useful, please cite Caffe paper:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
