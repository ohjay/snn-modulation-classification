#!/bin/bash

data="RadioML"
network_spec="networks/radio_ml_conv.yaml"
radio_ml_data_dir="/lif/radioml/2018.01/"

batch_size=96
batch_size_test=96
n_test_samples=480

python train.py \
    --data $data \
    --radio_ml_data_dir $radio_ml_data_dir \
    --network_spec $network_spec \
    --batch_size $batch_size \
    --batch_size_test $batch_size_test \
    --n_test_samples $n_test_samples
