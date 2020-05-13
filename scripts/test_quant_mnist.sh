#!/bin/bash

data="MNIST"
network_spec="networks/mnist_conv.yaml"
ref_network_spec="networks/mnist_conv.yaml"

restore_path="results/MNIST/012__04-05-2020-8-bit-weight-only/parameters_2500.pth"

forward_state_quantized="True"
batch_size=128
batch_size_test=512
n_test_samples=1024

weight_bit_width=8

python -u quant_test.py \
    --data $data \
    --network_spec $network_spec \
    --ref_network_spec $ref_network_spec \
    --batch_size $batch_size \
    --batch_size_test $batch_size_test \
    --n_test_samples $n_test_samples \
    --weight_bit_width $weight_bit_width \
    --restore_path $restore_path
    --forward_state_quantized $forward_state_quantized
