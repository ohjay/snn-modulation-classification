#!/bin/bash

radio_ml_data_dir="/media/owen/ba9d40b5-89de-4832-bad4-156b118e4a66/2018.01"
per_h5_frac=0.5
train_frac=0.9
I_resolution=16
Q_resolution=16
min_I=-1.0
max_I=1.0
min_Q=-1.0
max_Q=1.0

network_spec="networks/radio_ml_conv.yaml"
restore_path="results/RadioML/May27_12-00-41/parameters_80.pth"

burnin=900
n_iters_test=1024
batch_size_test=512
n_test_samples=512

python3 test_radio_ml.py \
    --radio_ml_data_dir $radio_ml_data_dir \
    --per_h5_frac $per_h5_frac \
    --train_frac $train_frac \
    --I_resolution $I_resolution \
    --Q_resolution $Q_resolution \
    --I_bounds $min_I $max_I \
    --Q_bounds $min_Q $max_Q \
    --network_spec $network_spec \
    --restore_path $restore_path \
    --burnin $burnin \
    --n_iters_test $n_iters_test \
    --batch_size_test $batch_size_test \
    --n_test_samples $n_test_samples
