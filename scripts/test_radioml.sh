#!/bin/bash

radio_ml_data_dir="/lif/radioml/2018.01/"

I_resolution=16
Q_resolution=16
min_I=-1.0
max_I=1.0
min_Q=-1.0
max_Q=1.0

network_spec="networks/radio_ml_conv.yaml"
restore_path="results/RadioML/Jun04_13-04-30/parameters_260.pth"
#restore_path="results/RadioML/May27_11-47-00/parameters_155.pth"


burnin=20
n_iters_test=1024
batch_size_test=512
n_test_samples=512
arp=1.0

python -u test_radioml.py \
    --radio_ml_data_dir $radio_ml_data_dir \
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
