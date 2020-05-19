#!/bin/bash

data="RadioML"

radio_ml_data_dir="/lif/radioml/2018.01/"

I_resolution=16
Q_resolution=16

min_I=-1.0
max_I=1.0
min_Q=-1.0
max_Q=1.0

network_spec="networks/radio_ml_conv.yaml"
ref_network_spec="networks/radio_ml_conv_ref.yaml"

burnin=50
batch_size=512
batch_size_test=512
n_test_samples=512

python -u quant_train.py \
    --data $data \
    --radio_ml_data_dir $radio_ml_data_dir \
    --I_resolution $I_resolution \
    --Q_resolution $Q_resolution \
    --I_bounds $min_I $max_I \
    --Q_bounds $min_Q $max_Q \
    --network_spec $network_spec \
    --ref_network_spec $ref_network_spec \
    --burnin $burnin \
    --batch_size $batch_size \
    --batch_size_test $batch_size_test \
    --n_test_samples $n_test_samples
