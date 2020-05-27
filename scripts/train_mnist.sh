#!/bin/bash

data="MNIST"
network_spec="networks/mnist_conv.yaml"
ref_network_spec="networks/mnist_conv.yaml"

burnin=50
n_iters=500
n_iters_test=1500
batch_size=128
batch_size_test=512
n_test_samples=1024

python -u train.py \
    --data $data \
    --network_spec $network_spec \
    --ref_network_spec $ref_network_spec \
    --burnin $burnin \
    --n_iters $n_iters \
    --n_iters_test $n_iters_test \
    --batch_size $batch_size \
    --batch_size_test $batch_size_test \
    --n_test_samples $n_test_samples
