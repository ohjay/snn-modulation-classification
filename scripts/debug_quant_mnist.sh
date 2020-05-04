#!/bin/bash

python -m pdb quant_train.py --data MNIST --network_spec networks/mnist_conv.yaml --batch_size 128 --batch_size_test 512 --n_test_samples 1024 --ref_network_spec networks/mnist_conv.yaml
