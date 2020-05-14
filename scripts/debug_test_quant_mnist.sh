#!/bin/bash

python -m pdb quant_test.py --data MNIST --network_spec networks/mnist_conv.yaml --batch_size 128 --batch_size_test 512 --n_test_samples 1024 --ref_network_spec networks/mnist_conv.yaml --weight_bit_width 8 --restore_path "results/MNIST/012__04-05-2020-8-bit-weight-only/parameters_2500.pth" --forward_state_quantized True --eps0_bit_width 16 --eps1_bit_width 16 --no_save True

