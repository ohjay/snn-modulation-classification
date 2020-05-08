import torch
import numpy as np
import os
import argparse
import time

from dcll.pytorch_libdcll import device
from networks import ConvNetwork, load_network_spec
from data.load_radio_ml import get_radio_ml_loader as get_loader
from data.utils import to_one_hot
from data.utils import iq2spiketrain as to_spike_train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--radio_ml_data_dir', type=str, default='2018.01',
                        help='path to the folder containing the RadioML HDF5 file(s)')
    parser.add_argument('--network_spec', type=str, default='networks/radio_ml_conv.yaml',
                        metavar='S', help='path to YAML file describing net architecture')
    parser.add_argument('--I_resolution', type=int, default=128,
                        metavar='N', help='size of I dimension (used when representing I/Q plane as image)')
    parser.add_argument('--Q_resolution', type=int, default=128,
                        metavar='N', help='size of Q dimension (used when representing I/Q plane as image)')
    parser.add_argument('--I_bounds', type=float, default=(-1, 1),
                        nargs=2, help='range of values to represent in I dimension of I/Q image')
    parser.add_argument('--Q_bounds', type=float, default=(-1, 1),
                        nargs=2, help='range of values to represent in Q dimension of I/Q image')
    parser.add_argument('--restore_path', type=str,
                        metavar='S', help='path to .pth file from which to restore')
    parser.add_argument('--burnin', type=int, default=50,
                        metavar='N', help='burnin')
    parser.add_argument('--batch_size_test', type=int, default=64,
                        metavar='N', help='input batch size for testing')
    parser.add_argument('--seed', type=int, default=1,
                        metavar='S', help='random seed')
    parser.add_argument('--n_test_samples', type=int, default=128,
                        metavar='N', help='how many test samples to use')
    parser.add_argument('--n_iters_test', type=int, default=1024, metavar='N',
                        help='for how many ms do we present a sample during classification')
    parser.add_argument('--alpha', type=float, default=.92,
                        metavar='N', help='Time constant for neuron')
    parser.add_argument('--alphas', type=float, default=.85,
                        metavar='N', help='Time constant for synapse')
    parser.add_argument('--alpharp', type=float, default=.65,
                        metavar='N', help='Time constant for refractory')
    parser.add_argument('--arp', type=float, default=0,
                        metavar='N', help='Absolute refractory period in ticks')
    parser.add_argument('--random_tau', type=bool, default=True,
                        help='randomize time constants in convolutional layers')
    parser.add_argument('--beta', type=float, default=.95,
                        metavar='N', help='Beta2 parameters for Adamax')
    parser.add_argument('--lc_ampl', type=float, default=.5,
                        metavar='N', help='magnitude of local classifier init')
    parser.add_argument('--netscale', type=float, default=1.,
                        metavar='N', help='scale network size')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    im_dims = (1, args.Q_resolution, args.I_resolution)
    target_size = 24
    # Set "to spike train" kwargs
    to_st_test_kwargs  = {
        'out_w': args.I_resolution,
        'out_h': args.Q_resolution,
        'min_I': args.I_bounds[0],
        'max_I': args.I_bounds[1],
        'min_Q': args.Q_bounds[0],
        'max_Q': args.Q_bounds[1],
        'max_duration': args.n_iters_test,
    }
    n_test = np.ceil(float(args.n_test_samples) / args.batch_size_test).astype(int)

    burnin = args.burnin
    convs = load_network_spec(args.network_spec)
    net = ConvNetwork(args, im_dims, args.batch_size_test, convs, target_size,
                    act=torch.nn.Sigmoid(), loss=None, opt=None, opt_param={},
                    learning_rates=None, burnin=burnin)

    if args.restore_path:
        print('-' * 80)
        if not os.path.isfile(args.restore_path):
            print('ERROR: Cannot load `%s`.' % args.restore_path)
            print('File does not exist! Aborting...')
            import sys; sys.exit(0)
        else:
            state_dict = torch.load(args.restore_path)
            net.load_state_dict(state_dict)
            print('Loaded the SNN model from `%s`.' % args.restore_path)
        print('-' * 80)

    net = net.to(device)
    net.reset(True)

    for snr in range(6, 32, 2):
        start_time = time.time()
        get_loader_kwargs  = {
            'data_dir': args.radio_ml_data_dir,
            'min_snr': snr,
            'max_snr': snr,
        }
        gen_test = iter(get_loader(args.batch_size_test, train=False, taskid=1, **get_loader_kwargs))
        all_test_data = [next(gen_test) for i in range(n_test)]
        all_test_data = [(samples, to_one_hot(labels, target_size))
                        for (samples, labels) in all_test_data]

        # Test
        acc_test = np.zeros([n_test, len(net.dcll_slices)])
        for i, test_data in enumerate(all_test_data):
            test_input, test_labels = to_spike_train(*test_data, **to_st_test_kwargs)
            try:
                test_input = torch.Tensor(test_input).to(device)
            except RuntimeError as e:
                print('Exception: ' + str(e) +
                        '. Try to decrease your batch_size_test with the --batch_size_test argument.')
                raise
            test_labels1h = torch.Tensor(test_labels).to(device)

            net.reset()
            net.eval()
            for sim_iteration in range(args.n_iters_test):
                net.test(x=test_input[sim_iteration])
            acc_test[i, :] = net.accuracy(test_labels1h)

        acc = np.mean(acc_test, axis=0)
        time_elapsed = (time.time() - start_time)
        time_elapsed = '%.2f s' % time_elapsed
        print('SNR {} \t Accuracy {} \t Time Elapsed {}'.format(str(snr).zfill(2), acc, time_elapsed))
