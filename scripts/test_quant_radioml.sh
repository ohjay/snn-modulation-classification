#!/bin/bash

data="RadioML"

radio_ml_data_dir="/lif/radioml/2018.01/"

restore_path="results/RadioML/Jun04_13-04-30/parameters_260.pth"

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

burnin=20
n_iters=1024
n_iters_test=1024
batch_size=512
batch_size_test=512
n_test_samples=512
learning_rates=(0.000000025)
ref_lr=0.001

weight_bit_width=32
eps0_bit_width=8
eps1_bit_width=8

forward_state_quantized="True"

arp=1.0

# -u to immediately print to stdout (for file redirect to see output before script ends)
python -u quant_test.py \
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
    --arp $arp \
    --n_iters_test $n_iters_test \
    --batch_size $batch_size \
    --batch_size_test $batch_size_test \
    --n_test_samples $n_test_samples \
    --n_test_interval 5 \
    --learning_rates "${learning_rates[@]/#/}" \
    --ref_lr $ref_lr \
    --weight_bit_width $weight_bit_width \
    --eps0_bit_width $eps0_bit_width \
    --eps1_bit_width $eps1_bit_width \
    --restore_path $restore_path \

    ##--forward_state_quantized $forward_state_quantized
