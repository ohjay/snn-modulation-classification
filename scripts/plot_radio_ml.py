import os
import math
import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt

from data.utils import to_one_hot
from data.utils import iq2spiketrain as to_spike_train
from data.load_radio_ml import get_radio_ml_loader as get_loader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--radio_ml_data_dir', type=str, default='2018.01',
                        help='path to the folder containing the RadioML HDF5 file(s)')
    parser.add_argument('--I_resolution', type=int, default=16,
                        metavar='N', help='size of I dimension (used when representing I/Q plane as image)')
    parser.add_argument('--Q_resolution', type=int, default=16,
                        metavar='N', help='size of Q dimension (used when representing I/Q plane as image)')
    parser.add_argument('--I_bounds', type=float, default=(-1, 1),
                        nargs=2, help='range of values to represent in I dimension of I/Q image')
    parser.add_argument('--Q_bounds', type=float, default=(-1, 1),
                        nargs=2, help='range of values to represent in Q dimension of I/Q image')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    get_loader_kwargs = {}
    to_st_test_kwargs = {}

    target_size = 24
    # Set "get loader" kwargs
    get_loader_kwargs['data_dir'] = args.radio_ml_data_dir
    # Set "to spike train" kwargs
    to_st_test_kwargs['out_w'] = args.I_resolution
    to_st_test_kwargs['out_h'] = args.Q_resolution
    to_st_test_kwargs['min_I'] = args.I_bounds[0]
    to_st_test_kwargs['max_I'] = args.I_bounds[1]
    to_st_test_kwargs['min_Q'] = args.Q_bounds[0]
    to_st_test_kwargs['max_Q'] = args.Q_bounds[1]
    to_st_test_kwargs['max_duration'] = 1024

    gen_test = iter(get_loader(64, train=False, taskid=1, **get_loader_kwargs))
    samples, labels = next(gen_test)
    labels_1h = to_one_hot(labels, target_size)

    st_samples, st_labels = to_spike_train(samples, labels_1h, **to_st_test_kwargs)

    b_idx = 52
    out_dir = 'plots'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with imageio.get_writer('spike_train.gif', mode='I', duration=0.1) as gif_writer:
        for t in range(1024):
            out_path = os.path.join(out_dir, 't%s.png' % str(t).zfill(4))
            plt.imshow(st_samples[t, b_idx, 0, :, :], extent=[-1, 1, 1, -1])
            plt.xlabel('I')
            plt.ylabel('Q')
            plt.savefig(out_path)
            plt.clf()
            if math.log(t + 1, 2).is_integer():
                print('Wrote `%s`.' % out_path)
            gif_writer.append_data(imageio.imread(out_path))
    print('Wrote `spike_train.gif`.')

    plt.figure(figsize=(20, 5))
    I_data, = plt.plot(np.arange(1024), samples[b_idx, 0, 0, :], 'r', label='I')
    Q_data, = plt.plot(np.arange(1024), samples[b_idx, 1, 0, :], 'g', label='Q')
    plt.legend(handles=[I_data, Q_data])
    plt.title('I/Q example for class %d' % labels[b_idx])
    plt.xlabel('time')
    plt.savefig('iq_plot.png')
    print('Wrote `iq_plot.png`.')
    plt.clf()
