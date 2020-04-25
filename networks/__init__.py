import torch
import numpy as np

from dcll.pytorch_libdcll import Conv2dDCLLlayer, DenseDCLLlayer, device, DCLLClassification


class ReferenceConvNetwork(torch.nn.Module):
    def __init__(self, args, im_dims, convs, loss, opt, opt_param, out_dim):
        super(ReferenceConvNetwork, self).__init__()

        def make_conv(inp, conf):
            layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=inp[0],
                                out_channels=int(conf[0] * args.netscale),
                                kernel_size=conf[1],
                                padding=conf[2]),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(
                    kernel_size=conf[3], stride=conf[3], padding=(conf[3]-1)//2)
            )
            layer = layer.to(device)
            return (layer, [conf[0]])

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
        self.linear = torch.nn.Linear(np.prod(latent_size()), out_dim).to(device)
        self.linear.weight.requires_grad = True
        self.linear.bias.requires_grad = True

        self.optim = opt(self.parameters(), **opt_param)
        self.crit = loss().to(device)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.linear(x.view(x.shape[0], -1))
        return x

    def learn(self, x, labels):
        y = self.forward(x)

        self.optim.zero_grad()
        loss = self.crit(y, labels)
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
                 loss, opt, opt_param,
                 DCLLSlice=DCLLClassification,
                 burnin=50
                 ):
        super(ConvNetwork, self).__init__()
        self.batch_size = batch_size

        def make_conv(inp, conf, is_output_layer=False):
            layer = Conv2dDCLLlayer(in_channels=inp[0], out_channels=int(conf[0] * args.netscale),
                                    kernel_size=conf[1], padding=conf[2], pooling=conf[3],
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
            name = 'conv%d' % i
            self.dcll_slices.append(
                DCLLSlice(
                    dclllayer=layer,
                    name=name,
                    batch_size=batch_size,
                    loss=loss,
                    optimizer=opt,
                    kwargs_optimizer=opt_param,
                    collect_stats=True,
                    burnin=burnin)
            )

    def learn(self, x, labels):
        spikes = x
        for sl in self.dcll_slices:
            spikes, _, _, _, _ = sl.train_dcll(
                spikes, labels, regularize=False)

    def test(self, x):
        spikes = x
        for sl in self.dcll_slices:
            spikes, _, _, _ = sl.forward(spikes)

    def reset(self, init_states=False):
        [s.init(self.batch_size, init_states=init_states)
         for s in self.dcll_slices]

    def write_stats(self, writer, epoch, comment=''):
        [s.write_stats(writer, label='test'+comment+'/', epoch=epoch)
         for s in self.dcll_slices]

    def accuracy(self, labels):
        return [s.accuracy(labels) for s in self.dcll_slices]
