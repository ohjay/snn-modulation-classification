import torch
import pickle
import numpy as np
import os
import argparse
import datetime
from tensorboardX import SummaryWriter

import subprocess # execute git command

from dcll.pytorch_libdcll import device
from dcll.experiment_tools import mksavedir, save_source, annotate
from dcll.pytorch_utils import grad_parameters, named_grad_parameters, NetworkDumper, tonumpy
from networks import ConvNetwork, ReferenceConvNetwork, load_network_spec
from data.utils import to_one_hot

import time

def parse_args():
    parser = argparse.ArgumentParser(description='DCLL')
    parser.add_argument('--data', type=str, default='RadioML',
                        choices=['MNIST', 'RadioML'], help='which data to use')
    parser.add_argument('--radio_ml_data_dir', type=str, default='2018.01',
                        help='path to the folder containing the RadioML HDF5 file(s)')
    parser.add_argument('--network_spec', type=str, default='networks/radio_ml_conv.yaml',
                        metavar='S', help='path to YAML file describing net architecture')
    parser.add_argument('--ref_network_spec', type=str, default='networks/radio_ml_conv_ref.yaml',
                        metavar='S', help='path to YAML file describing reference net architecture')
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
    parser.add_argument('--batch_size', type=int, default=64,
                        metavar='N', help='input batch size for training')
    parser.add_argument('--batch_size_test', type=int, default=64,
                        metavar='N', help='input batch size for testing')
    parser.add_argument('--n_steps', type=int, default=10000,
                        metavar='N', help='number of steps to train')
    parser.add_argument('--no_save', type=bool, default=False,
                        metavar='N', help='disables saving into Results directory')
    parser.add_argument('--seed', type=int, default=1,
                        metavar='S', help='random seed')
    parser.add_argument('--n_test_interval', type=int, default=20,
                        metavar='N', help='how many steps to run before testing')
    parser.add_argument('--n_test_samples', type=int, default=128,
                        metavar='N', help='how many test samples to use')
    parser.add_argument('--n_iters', type=int, default=500, metavar='N',
                        help='for how many ms do we present a sample during classification')
    parser.add_argument('--n_iters_test', type=int, default=1500, metavar='N',
                        help='for how many ms do we present a sample during classification')
    parser.add_argument('--optim_type', type=str, default='Adamax',
                        metavar='S', help='which optimizer to use')
    parser.add_argument('--loss_type', type=str, default='SmoothL1Loss',
                        metavar='S', help='which loss function to use')
    parser.add_argument('--lr', type=float, default=2.5e-8,
                        metavar='N', help='learning rate for Adamax')
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
    parser.add_argument('--comment', type=str, default='',
                        help='comment to name tensorboard files')
    parser.add_argument('--output', type=str, default='results',
                        help='folder name for the results')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', args.data, current_time)
    print('log dir: {log_dir}'.format(log_dir=log_dir))

    get_loader_kwargs  = {}
    to_st_train_kwargs = {}
    to_st_test_kwargs  = {}

    if args.data == 'MNIST':
        im_dims = (1, 28, 28)
        ref_im_dims = (1, 28, 28)
        target_size = 10
        from data.load_mnist import get_mnist_loader as get_loader
        from data.utils import image2spiketrain as to_spike_train
        # Set "to spike train" kwargs
        n_iters = args.n_iters
        n_iters_test = args.n_iters_test
        for to_st_kwargs in (to_st_train_kwargs, to_st_test_kwargs):
            to_st_kwargs['input_shape'] = im_dims
            to_st_kwargs['gain'] = 100
        to_st_train_kwargs['min_duration'] = n_iters - 1
        to_st_train_kwargs['max_duration'] = n_iters
        to_st_test_kwargs['min_duration'] = n_iters_test - 1
        to_st_test_kwargs['max_duration'] = n_iters_test

    elif args.data == 'RadioML':
        im_dims = (1, args.Q_resolution, args.I_resolution)
        ref_im_dims = (2, 1, 1024)
        target_size = 24
        from data.load_radio_ml import get_radio_ml_loader as get_loader
        from data.utils import iq2spiketrain as to_spike_train
        # Set "get loader" kwargs
        get_loader_kwargs['data_dir'] = args.radio_ml_data_dir
        # Set "to spike train" kwargs
        n_iters = 1024
        n_iters_test = 1024
        for to_st_kwargs in (to_st_train_kwargs, to_st_test_kwargs):
            to_st_kwargs['out_w'] = args.I_resolution
            to_st_kwargs['out_h'] = args.Q_resolution
            to_st_kwargs['min_I'] = args.I_bounds[0]
            to_st_kwargs['max_I'] = args.I_bounds[1]
            to_st_kwargs['min_Q'] = args.Q_bounds[0]
            to_st_kwargs['max_Q'] = args.Q_bounds[1]
            to_st_kwargs['max_duration'] = n_iters

    # number of test samples: n_test * batch_size_test
    n_test = np.ceil(float(args.n_test_samples) /
                     args.batch_size_test).astype(int)

    opt = getattr(torch.optim, args.optim_type)
    opt_param = {'lr': args.lr, 'betas': [.0, args.beta]}
    loss = getattr(torch.nn, args.loss_type)

    burnin = args.burnin
    convs = load_network_spec(args.network_spec)
    net = ConvNetwork(args, im_dims, args.batch_size, convs, target_size,
                      act=torch.nn.Sigmoid(), loss=loss, opt=opt, opt_param=opt_param, burnin=burnin)

    if args.restore_path:
        print('-' * 80)
        if not os.path.isfile(args.restore_path):
            print('ERROR: Cannot load `%s`.' % args.restore_path)
            print('File does not exist! Aborting load...')
        else:
            state_dict = torch.load(args.restore_path)
            net.load_state_dict(state_dict)
            print('Loaded the SNN model from `%s`.' % args.restore_path)
        print('-' * 80)

    net = net.to(device)
    net.reset(True)

    ref_convs = load_network_spec(args.ref_network_spec)
    ref_net = ReferenceConvNetwork(args, ref_im_dims, ref_convs, loss, opt, opt_param, target_size)
    ref_net = ref_net.to(device)

    writer = SummaryWriter(log_dir=log_dir, comment='%s Conv' % args.data)
    dumper = NetworkDumper(writer, net)

    if not args.no_save:
        out_dir = os.path.join(args.output, args.data)
        d = mksavedir(pre=out_dir)
        annotate(d, text=log_dir, filename='log_filename.txt')
        annotate(d, text=str(args), filename='args.txt')
        with open(os.path.join(d, 'args.pkl'), 'wb') as fp:
            pickle.dump(vars(args), fp)
        save_source(d)

        print(args)

        # Log python commit
        commit_logfile = open(os.path.join(d, 'current_commit.txt'), 'w')
        commit_msg = subprocess.check_output('git log -1', shell=True)
        commit_logfile.write(commit_msg.decode('utf-8'))
        commit_logfile.close()

    n_tests_total = np.ceil(float(args.n_steps) /
                            args.n_test_interval).astype(int)
    acc_test = np.empty([n_tests_total, n_test, len(net.dcll_slices)])
    acc_test_ref = np.empty([n_tests_total, n_test])

    train_data = get_loader(args.batch_size, train=True, taskid=0, **get_loader_kwargs)
    gen_train = iter(train_data)
    gen_test = iter(get_loader(args.batch_size_test, train=False, taskid=1, **get_loader_kwargs))

    all_test_data = [next(gen_test) for i in range(n_test)]
    all_test_data = [(samples, to_one_hot(labels, target_size))
                     for (samples, labels) in all_test_data]

    print("\n\nStart Training\n")

    for step in range(args.n_steps):
        print("Minibatch step {}".format(step))
        start = time.time()
        if ((step + 1) % 1000) == 0:
            for i in range(len(net.dcll_slices)):
                net.dcll_slices[i].optimizer.param_groups[-1]['lr'] /= 2
            net.dcll_slices[-1].optimizer2.param_groups[-1]['lr'] /= 2
            ref_net.optim.param_groups[-1]['lr'] /= 2
            print('Adjusting learning rates')

        try:
            input, labels = next(gen_train)
        except StopIteration:
            gen_train = iter(train_data)
            input, labels = next(gen_train)
        labels = to_one_hot(labels, target_size)

        print("Before to_spike_train {}".format(time.time() - start))
        input_spikes, labels_spikes = to_spike_train(input, labels,
                                                     **to_st_train_kwargs)
        input_spikes = torch.Tensor(input_spikes).to(device)
        labels_spikes = torch.Tensor(labels_spikes).to(device)

        ref_input = torch.Tensor(input).to(device).reshape(-1, *ref_im_dims)
        ref_label = torch.Tensor(labels).to(device)

        print("Before train {}".format(time.time() - start))
        # Train
        net.reset()
        net.train()
        ref_net.train()
        for sim_iteration in range(n_iters):
            if (sim_iteration % 50 == 0):
                print("Iteration {} time: {}:".format(sim_iteration, time.time() - start))
            net.learn(x=input_spikes[sim_iteration],
                      labels=labels_spikes[sim_iteration])
            ref_net.learn(x=ref_input, labels=ref_label)

        acc = net.accuracy(labels_spikes)
        print("Step {} Training accuracy: {}".format(step, acc))

        # Test
        if (step % args.n_test_interval) == 0:
            test_idx = step // args.n_test_interval
            for i, test_data in enumerate(all_test_data):
                test_input, test_labels = to_spike_train(*test_data,
                                                         **to_st_test_kwargs)
                try:
                    test_input = torch.Tensor(test_input).to(device)
                except RuntimeError as e:
                    print('Exception: ' + str(e) +
                          '. Try to decrease your batch_size_test with the --batch_size_test argument.')
                    raise

                test_labels1h = torch.Tensor(test_labels).to(device)
                test_ref_input = torch.Tensor(test_data[0]).to(device).reshape(-1, *ref_im_dims)
                test_ref_label = torch.Tensor(test_data[1]).to(device)

                net.reset()
                net.eval()
                ref_net.eval()
                for sim_iteration in range(n_iters_test):
                    net.test(x=test_input[sim_iteration])

                ref_net.test(test_ref_input)

                acc_test[test_idx, i, :] = net.accuracy(test_labels1h)
                acc_test_ref[test_idx, i] = ref_net.accuracy(test_ref_label)

                if i == 0:
                    net.write_stats(writer, step, comment='_batch_'+str(i))
                    ref_net.write_stats(writer, step)

            if not args.no_save:
                np.save(d + '/acc_test.npy', acc_test)
                np.save(d + '/acc_test_ref.npy', acc_test_ref)

                # Save network parameters
                save_path = os.path.join(d, 'parameters_{}.pth'.format(step))
                torch.save(net.cpu().state_dict(), save_path)
                net = net.to(device)
                print('-' * 80)
                print('Saved network parameters to `%s`.' % save_path)
                print('-' * 80)

            acc = np.mean(acc_test[test_idx], axis=0)
            acc_ref = np.mean(acc_test_ref[test_idx], axis=0)
            print('Step {} \t Accuracy {} \t Ref {}'.format(step, acc, acc_ref))
            logfile = open(os.path.join(d, 'logfile-accuracy.txt'), 'a')
            logfile.write('Step {} \t Accuracy {} \t Ref {}\n'.format(step, acc, acc_ref))
            logfile.close()

    writer.close()
