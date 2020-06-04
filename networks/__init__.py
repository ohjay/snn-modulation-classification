import yaml
import torch
import numpy as np
from ast import literal_eval as make_tuple

from dcll.pytorch_libdcll import Conv2dDCLLlayer, DenseDCLLlayer, device, DCLLClassification


def load_network_spec(yaml_path):
    network_spec = yaml.load(open(yaml_path, 'r'))
    convs = network_spec['conv_layers']
    for layer_idx, layer_spec in enumerate(convs):
        for k, v in layer_spec.items():
            if type(v) != int:
                # convert e.g. the string "(2, 0)" to the tuple (2, 0)
                convs[layer_idx][k] = make_tuple(v)
    return convs


class ReferenceConvNetwork(torch.nn.Module):
    def __init__(self, args, im_dims, convs, loss, opt, opt_param, out_dim):
        super(ReferenceConvNetwork, self).__init__()

        def make_conv(inp, conf):
            out_channels = conf['out_channels']
            kernel_size  = conf['kernel_size']
            padding      = conf['padding']
            pooling      = conf['pooling']

            pool_pad = None
            if type(pooling) == int:
                pool_pad = (pooling - 1) // 2
            elif type(pooling) == tuple:
                pool_pad = tuple((np.array(pooling) - 1) // 2)
            else:
                raise ValueError('unsupported pooling spec type: %r' % type(pooling))

            layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=inp[0],
                                out_channels=int(out_channels * args.netscale),
                                kernel_size=kernel_size,
                                padding=padding),
                torch.nn.MaxPool2d(
                    kernel_size=pooling, stride=pooling, padding=pool_pad),
                torch.nn.BatchNorm2d(int(out_channels * args.netscale)),
                torch.nn.ReLU()
            )
            layer = layer.to(device)
            return (layer, [out_channels])

        n = im_dims
        self.num_layers = len(convs)
        self.layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            layer, n = make_conv(n, convs[i])
            self.layers.append(layer)

        def latent_size():
            with torch.no_grad():
                x = torch.zeros(im_dims).unsqueeze(0).to(device)
                for layer in self.layers:
                    x = layer(x)
            return x.shape[1:]

        # Should we train linear decoders? They are not in DCLL
        self.linear1 = torch.nn.Sequential(
            torch.nn.Linear(np.prod(latent_size()), 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.SELU()
        ).to(device)

        self.linear2 = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.SELU()
        ).to(device)

        self.linear3 = torch.nn.Sequential(
            torch.nn.Linear(128, out_dim),
            torch.nn.Softmax()
        ).to(device)

        self.optim = opt(self.parameters(), **opt_param)
        self.crit = torch.nn.NLLLoss().to(device)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

    def learn(self, x, labels):
        y = self.forward(x)

        self.optim.zero_grad()
        loss = self.crit(y, labels.argmax(1))
        loss.backward()
        self.optim.step()

    def test(self, x):
        self.y_test = self.forward(x.detach())

    def write_stats(self, writer, epoch):
        writer.add_scalar('acc/ref_net', self.acc, epoch)

    def accuracy(self, labels):
        self.acc = torch.mean(
            (self.y_test.argmax(1) == labels.argmax(1)).float()).item()
        return self.acc


class ConvNetwork(torch.nn.Module):
    def __init__(self, args, im_dims, batch_size, convs,
                 target_size, act,
                 loss, opt, opt_param, learning_rates,
                 DCLLSlice=DCLLClassification,
                 burnin=50
                 ):
        super(ConvNetwork, self).__init__()
        self.batch_size = batch_size

        def make_conv(inp, conf, is_output_layer=False):
            out_channels = conf['out_channels']
            kernel_size  = conf['kernel_size']
            padding      = conf['padding']
            pooling      = conf['pooling']

            layer = Conv2dDCLLlayer(in_channels=inp[0],
                                    out_channels=int(out_channels * args.netscale),
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    pooling=pooling,
                                    im_dims=inp[1:3],  # height, width
                                    target_size=target_size,
                                    alpha=args.alpha, alphas=args.alphas, alpharp=args.alpharp,
                                    wrp=args.arp, act=act, lc_ampl=args.lc_ampl,
                                    random_tau=args.random_tau,
                                    spiking=True,
                                    lc_dropout=.5,
                                    output_layer=is_output_layer
                                    ).to(device).init_hiddens(batch_size)
            return layer, torch.Size([layer.out_channels]) + layer.output_shape

        n = im_dims
        self.num_layers = len(convs)
        self.layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            is_output_layer = (i == self.num_layers - 1)
            layer, n = make_conv(n, convs[i], is_output_layer)
            self.layers.append(layer)

        self.dcll_slices = []
        for i, layer in enumerate(self.layers):
            layer_opt_param = opt_param.copy()
            if learning_rates is not None:
                lr_idx = min(i, len(learning_rates) - 1)
                layer_opt_param['lr'] = learning_rates[lr_idx]
            name = 'conv%d' % i
            self.dcll_slices.append(
                DCLLSlice(
                    dclllayer=layer,
                    name=name,
                    batch_size=batch_size,
                    loss=loss,
                    optimizer=opt,
                    kwargs_optimizer=layer_opt_param,
                    collect_stats=True,
                    burnin=burnin)
            )

    def learn(self, x, labels):
        spikes = x
        for s in self.dcll_slices:
            spikes, _, _, _, _ = s.train_dcll(
                spikes, labels, regularize=False)

    def test(self, x):
        spikes = x
        for s in self.dcll_slices:
            spikes, _, _, _ = s.forward(spikes)

    def reset(self, init_states=False):
        for s in self.dcll_slices:
            s.init(self.batch_size, init_states=init_states)

    def write_stats(self, writer, epoch, comment=''):
        for s in self.dcll_slices:
            s.write_stats(writer, label='test'+comment, epoch=epoch)

    def accuracy(self, labels):
        return [s.accuracy(labels) for s in self.dcll_slices]

    def predictions(self):
        return [s.predictions() for s in self.dcll_slices]
