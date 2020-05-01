#!/bin/bash

data="MNIST"
network_spec="networks/mnist_conv.yaml"

batch_size=128
batch_size_test=512
n_test_samples=1024

echo "train mnist" >> /lif/log2

# -u to immediately print to stdout (for file redirect to see output before script ends)
python train.py -u \
    --data $data \
    --network_spec $network_spec \
    --batch_size $batch_size \
    --batch_size_test $batch_size_test \
    --n_test_samples $n_test_samples > /lif/log-train
