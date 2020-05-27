#!/bin/bash

data="RadioML"
radio_ml_data_dir="/media/owen/ba9d40b5-89de-4832-bad4-156b118e4a66/2018.01"
min_snr=6
max_snr=30
per_h5_frac=0.5
train_frac=0.9
I_resolution=16
Q_resolution=16
min_I=-1.0
max_I=1.0
min_Q=-1.0
max_Q=1.0

network_spec="networks/radio_ml_conv.yaml"
ref_network_spec="networks/radio_ml_conv_ref.yaml"

burnin=900
n_iters=1024
n_iters_test=1024
batch_size=512
batch_size_test=512
n_test_samples=512
n_test_interval=10
learning_rates=(0.000000025)
ref_lr=0.001

python3 train.py \
    --data $data \
    --radio_ml_data_dir $radio_ml_data_dir \
    --min_snr $min_snr \
    --max_snr $max_snr \
    --per_h5_frac $per_h5_frac \
    --train_frac $train_frac \
    --I_resolution $I_resolution \
    --Q_resolution $Q_resolution \
    --I_bounds $min_I $max_I \
    --Q_bounds $min_Q $max_Q \
    --network_spec $network_spec \
    --ref_network_spec $ref_network_spec \
    --burnin $burnin \
    --n_iters $n_iters \
    --n_iters_test $n_iters_test \
    --batch_size $batch_size \
    --batch_size_test $batch_size_test \
    --n_test_samples $n_test_samples \
    --n_test_interval $n_test_interval \
    --learning_rates "${learning_rates[@]/#/}" \
    --ref_lr $ref_lr
