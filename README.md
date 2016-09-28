# Caffe Framework with various triplet loss neural network
This is my caffe framework which is initially designed for triplet loss neural network. It contains various useful implementations, including:

1. create LEVELDB/LMDB for triplet input: anchor/postive/negative. Code in **tools/convert_triplet_db_dataset.cpp**.
2. create LEVELDB/LMDB for multiple postive and negative input. Code in **tools/convert_multiple_triplet_db_dataset.cpp**.
3. loss layer with multiple postive and negative input. Code in **src/caffe/layers/naive_triplet_multiple_loss_layer.cpp**.
