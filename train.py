import torch
import pickle
import numpy as np
import os
import argparse

from dcll.pytorch_libdcll import device
from dcll.experiment_tools import mksavedir, save_source, annotate
from dcll.pytorch_utils import grad_parameters, named_grad_parameters, NetworkDumper, tonumpy

from networks import ConvNetwork, ReferenceConvNetwork, load_network_spec


def parse_args():
    parser = argparse.ArgumentParser(description='DCLL')
    parser.add_argument('--data', type=str, default='RadioML',
                        choices=['MNIST', 'RadioML'], help='which data to use')
    parser.add_argument('--network_spec', type=str, default='networks/radio_ml_conv.yaml',
                        metavar='S', help='path to YAML file describing net architecture')
    parser.add_argument('--batch_size', type=int, default=64,
                        metavar='N', help='input batch size for training')
    parser.add_argument('--batch_size_test', type=int, default=64,
                        metavar='N', help='input batch size for testing')
    parser.add_argument('--n_epochs', type=int, default=10000,
                        metavar='N', help='number of epochs to train')
    parser.add_argument('--no_save', type=bool, default=False,
                        metavar='N', help='disables saving into Results directory')
    parser.add_argument('--seed', type=int, default=1,
                        metavar='S', help='random seed')
    parser.add_argument('--n_test_interval', type=int, default=20,
                        metavar='N', help='how many epochs to run before testing')
    parser.add_argument('--n_test_samples', type=int, default=128,
                        metavar='N', help='how many test samples to use')
    parser.add_argument('--n_iters_test', type=int, default=1500, metavar='N',
                        help='for how many ms do we present a sample during classification')
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

    import datetime
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', args.data, current_time)
    print('log dir: {log_dir}'.format(log_dir=log_dir))

    if args.data == 'MNIST':
        im_dims = (1, 28, 28)
        target_size = 10
        from data.load_mnist import get_mnist_loader as get_loader

    elif args.data == 'RadioML':
        im_dims = (2, 1, 1024)
        target_size = 24
        from data.load_radio_ml import get_radio_ml_loader as get_loader

    n_iters = 500
    n_iters_test = args.n_iters_test
    # number of test samples: n_test * batch_size_test
    n_test = np.ceil(float(args.n_test_samples) /
                     args.batch_size_test).astype(int)

    opt = torch.optim.Adamax
    opt_param = {'lr': args.lr, 'betas': [.0, args.beta]}
    loss = getattr(torch.nn, args.loss_type)

    burnin = 50
    convs = load_network_spec(args.network_spec)
    net = ConvNetwork(args, im_dims, args.batch_size, convs, target_size,
                      act=torch.nn.Sigmoid(),
                      loss=loss, opt=opt, opt_param=opt_param, burnin=burnin)
    net.reset(True)

    ref_net = ReferenceConvNetwork(args, im_dims, convs, loss, opt, opt_param, target_size)

    from tensorboardX import SummaryWriter
    writer = SummaryWriter(log_dir=log_dir, comment='%s Conv' % args.data)
    dumper = NetworkDumper(writer, net)

    if not args.no_save:
        out_dir = os.path.join(args.output, args.data)
        d = mksavedir(pre=out_dir)
        annotate(d, text=log_dir, filename='log_filename')
        annotate(d, text=str(args), filename='args')
        with open(os.path.join(d, 'args.pkl'), 'wb') as fp:
            pickle.dump(vars(args), fp)
        save_source(d)

    n_tests_total = np.ceil(float(args.n_epochs) /
                            args.n_test_interval).astype(int)
    acc_test = np.empty([n_tests_total, n_test, len(net.dcll_slices)])
    acc_test_ref = np.empty([n_tests_total, n_test])

    from data.utils import to_one_hot, image2spiketrain
    train_data = get_loader(args.batch_size, train=True, taskid=0)
    gen_train = iter(train_data)
    gen_test = iter(get_loader(args.batch_size_test, train=False, taskid=1))

    all_test_data = [next(gen_test) for i in range(n_test)]
    all_test_data = [(samples, to_one_hot(labels, target_size))
                     for (samples, labels) in all_test_data]

    for epoch in range(args.n_epochs):
        if ((epoch + 1) % 1000) == 0:
            net.dcll_slices[0].optimizer.param_groups[-1]['lr'] /= 2
            net.dcll_slices[1].optimizer.param_groups[-1]['lr'] /= 2
            net.dcll_slices[2].optimizer.param_groups[-1]['lr'] /= 2
            net.dcll_slices[2].optimizer2.param_groups[-1]['lr'] /= 2
            ref_net.optim.param_groups[-1]['lr'] /= 2
            print('Adjusting learning rates')

        try:
            input, labels = next(gen_train)
        except StopIteration:
            gen_train = iter(train_data)
            input, labels = next(gen_train)

        labels = to_one_hot(labels, target_size)

        input_spikes, labels_spikes = image2spiketrain(input, labels,
                                                       input_shape=im_dims,
                                                       target_size=target_size,
                                                       min_duration=n_iters-1,
                                                       max_duration=n_iters,
                                                       gain=100)
        input_spikes = torch.Tensor(input_spikes).to(device)
        labels_spikes = torch.Tensor(labels_spikes).to(device)

        ref_input = torch.Tensor(input).to(device).reshape(-1, *im_dims)
        ref_label = torch.Tensor(labels).to(device)

        net.reset()
        # Train
        net.train()
        ref_net.train()
        for sim_iteration in range(n_iters):
            net.learn(x=input_spikes[sim_iteration],
                      labels=labels_spikes[sim_iteration])
            ref_net.learn(x=ref_input, labels=ref_label)

        if (epoch % args.n_test_interval) == 0:
            for i, test_data in enumerate(all_test_data):
                test_input, test_labels = image2spiketrain(*test_data,
                                                           input_shape=im_dims,
                                                           target_size=target_size,
                                                           min_duration=n_iters_test-1,
                                                           max_duration=n_iters_test,
                                                           gain=100)
                try:
                    test_input = torch.Tensor(test_input).to(device)
                except RuntimeError as e:
                    print('Exception: ' + str(e) +
                          '. Try to decrease your batch_size_test with the --batch_size_test argument.')
                    raise

                test_labels1h = torch.Tensor(test_labels).to(device)
                test_ref_input = torch.Tensor(test_data[0]).to(
                    device).reshape(-1, *im_dims)
                test_ref_label = torch.Tensor(test_data[1]).to(device)

                net.reset()
                net.eval()
                ref_net.eval()
                # Test
                for sim_iteration in range(n_iters_test):
                    net.test(x=test_input[sim_iteration])

                ref_net.test(test_ref_input)

                acc_test[epoch//args.n_test_interval,
                         i, :] = net.accuracy(test_labels1h)
                acc_test_ref[epoch//args.n_test_interval,
                             i] = ref_net.accuracy(test_ref_label)

                if i == 0:
                    net.write_stats(writer, epoch, comment='_batch_'+str(i))
                    ref_net.write_stats(writer, epoch)
            if not args.no_save:
                np.save(d+'/acc_test.npy', acc_test)
                np.save(d+'/acc_test_ref.npy', acc_test_ref)
                annotate(d, text='', filename='best result')
                parameter_dict = {
                    name: data.detach().cpu().numpy()
                    for (name, data) in net.named_parameters()
                }
                with open(d+'/parameters_{}.pkl'.format(epoch), 'wb') as f:
                    pickle.dump(parameter_dict, f)
            print('Epoch {} \t Accuracy {} \t Ref {}'.format(epoch, np.mean(
                acc_test[epoch//args.n_test_interval], axis=0), np.mean(acc_test_ref[epoch//args.n_test_interval], axis=0)))

    writer.close()
