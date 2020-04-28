#!/bin/bash

data="RadioML"
network_spec="networks/radio_ml_conv.yaml"
ref_network_spec="networks/radio_ml_conv_ref.yaml"
radio_ml_data_dir="/media/owen/ba9d40b5-89de-4832-bad4-156b118e4a66/2018.01"

batch_size=256
batch_size_test=256
n_test_samples=512

python train.py \
    --data $data \
    --radio_ml_data_dir $radio_ml_data_dir \
    --network_spec $network_spec \
    --ref_network_spec $ref_network_spec \
    --batch_size $batch_size \
    --batch_size_test $batch_size_test \
    --n_test_samples $n_test_samples
