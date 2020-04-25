#!/bin/bash

data="RadioML"

batch_size=96
batch_size_test=96
n_test_samples=480

python train.py \
    --data $data \
    --batch_size $batch_size \
    --batch_size_test $batch_size_test \
    --n_test_samples $n_test_samples
