import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.nn import functional as F
import numpy as np
from collections import namedtuple
import logging
from collections import Counter
import math

from dcll.pytorch_libdcll import device

class QuantContinuousConv2D(nn.Module):
    NeuronState = namedtuple('NeuronState', ('eps0', 'eps1'))

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=2,
                 dilation=1,
                 groups=1,
                 bias=True,
                 alpha=.95,
                 alphas=.9,
                 act=nn.Sigmoid(),
                 random_tau=False,
                 spiking=True,
                 **kwargs):
        super(QuantContinuousConv2D, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(padding, int):
            padding = (padding, padding)

        self.random_tau = random_tau
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.act = act
        self.spiking = spiking
        if spiking:
            self.output_act = lambda x: (x > 0).float()
        else:
            self.output_act = lambda x: x

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.alpha = torch.nn.Parameter(
            torch.Tensor([alpha]), requires_grad=False)
        self.tau_m__dt = torch.nn.Parameter(
            torch.Tensor([1./(1-self.alpha)]), requires_grad=False)
        self.alphas = torch.nn.Parameter(
            torch.Tensor([alphas]), requires_grad=False)
        self.tau_s__dt = torch.nn.Parameter(torch.Tensor(
            [1./(1-self.alphas)]), requires_grad=False)
        print(act)

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n) / 250
        self.weight.data.uniform_(-stdv*1e-2, stdv*1e-2)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def get_output_shape(self, im_dims):
        im_height = im_dims[0]
        im_width = im_dims[1]
        height = (
            (im_height+2*self.padding[0]-self.dilation*(self.kernel_size[0]-1)-1)//self.stride+1)
        weight = (
            (im_width+2*self.padding[1]-self.dilation*(self.kernel_size[1]-1)-1)//self.stride+1)
        return height, weight

    def init_state(self, batch_size, im_dims, init_value=0):
        input_shape = [batch_size, self.in_channels, im_dims[0], im_dims[1]]

        self.state = self.NeuronState(
            eps0=torch.zeros(input_shape).detach().to(device)+init_value,
            eps1=torch.zeros(input_shape).detach().to(device)+init_value
        )

        if self.random_tau:
            self.randomize_tau(im_dims)
            self.random_tau = False

        return self.state

    def randomize_tau(self, im_dims, low=[5, 5], high=[10, 35]):
        taum = np.random.uniform(low[1], high[1], size=[self.in_channels])*1e-3
        taus = np.random.uniform(low[0], high[0], size=[self.in_channels])*1e-3
        taum = np.broadcast_to(
            taum, (im_dims[0], im_dims[1], self.in_channels)).transpose(2, 0, 1)
        taus = np.broadcast_to(
            taus, (im_dims[0], im_dims[1], self.in_channels)).transpose(2, 0, 1)
        self.alpha = torch.nn.Parameter(torch.Tensor(
            1-1e-3/taum).to(device), requires_grad=False)
        self.tau_m__dt = torch.nn.Parameter(
            1./(1-self.alpha), requires_grad=False)
        self.alphas = torch.nn.Parameter(torch.Tensor(
            1-1e-3/taus).to(device), requires_grad=False)
        self.tau_s__dt = torch.nn.Parameter(
            1./(1-self.alphas), requires_grad=False)

    def forward(self, input):
        # input: input tensor of shape (minibatch x in_channels x iH x iW)
        # weight: filters of shape (out_channels x (in_channels / groups) x kH x kW)
        if not (input.shape[0] == self.state.eps0.shape[0] == self.state.eps1.shape[0]):
            logging.warning("Batch size changed from {} to {} since last iteration. Reallocating states."
                            .format(self.state.eps0.shape[0], input.shape[0]))
            self.init_state(input.shape[0], input.shape[2:4])

        eps0 = input * self.tau_s__dt + self.alphas * self.state.eps0
        eps1 = self.alpha * self.state.eps1 + eps0 * self.tau_m__dt
        pvmem = F.conv2d(eps1, self.weight, self.bias, self.stride,
                         self.padding, self.dilation, self.groups)
        pv = self.act(pvmem)
        output = self.output_act(pvmem)

        # best
        #arp = .65*self.state.arp + output*10
        self.state = self.NeuronState(eps0=eps0.detach(),
                                      eps1=eps1.detach())
        return output, pv, pvmem

    def init_prev(self, batch_size, im_dims):
        return torch.zeros(batch_size, self.in_channels, im_dims[0], im_dims[1])


class QuantContinuousRelativeRefractoryConv2D(QuantContinuousConv2D):
    NeuronState = namedtuple('NeuronState', ('eps0', 'eps1', 'arp'))

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=2,
                 dilation=1,
                 groups=1,
                 bias=True,
                 alpha=.95,
                 alphas=.9,
                 alpharp=.65,
                 wrp=1,
                 act=nn.Sigmoid(),
                 random_tau=False,
                 **kwargs):
        """
        Continuous local learning with relative refractory period. No isyn or vmem dynamics for speed and memory.
        *wrp*: weight for the relative refractory period
        """
        super(QuantContinuousRelativeRefractoryConv2D, self).__init__(in_channels, out_channels,
                                                                 kernel_size, stride, padding, dilation, groups, bias, alpha, alphas, act)

        print("Relative RP")
        # best
        # self.tarp=10
        self.wrp = wrp
        self.alpharp = alpharp
        self.tau_rp__dt = 1./(1-self.alpharp)
        self.iter = 0
        self.tau_set = False
        self.random_tau = random_tau

    def init_state(self, batch_size, im_dims, init_value=0):
        input_shape = [batch_size, self.in_channels, im_dims[0], im_dims[1]]
        output_shape = torch.Size(
            [batch_size, self.out_channels]) + self.get_output_shape(im_dims)

        self.state = self.NeuronState(
            eps0=torch.zeros(input_shape).to(device)+init_value,
            eps1=torch.zeros(input_shape).to(device)+init_value,
            arp=torch.zeros(output_shape).to(device),
        )

        if self.random_tau:
            self.randomize_tau(im_dims)
            self.random_tau = True

        return self.state

    def forward(self, input):
        # input: input tensor of shape (minibatch x in_channels x iH x iW)
        # weight: filters of shape (out_channels x (in_channels / groups) x kH x kW)
        if not (input.shape[0] == self.state.eps0.shape[0] == self.state.eps1.shape[0]):
            logger.warning("Batch size changed from {} to {} since last iteration. Reallocating states."
                           .format(self.state.eps0.shape[0], input.shape[0]))
            self.init_state(input.shape[0], input.shape[2:4])

        eps0 = input*self.tau_s__dt + self.alphas * self.state.eps0
        eps1 = self.alpha * self.state.eps1 + eps0*self.tau_m__dt
        pvmem = F.conv2d(eps1, self.weight, self.bias, self.stride,
                         self.padding, self.dilation, self.groups)
        arp = self.alpharp*self.state.arp
        outpvmem = pvmem+arp
        output = (outpvmem > 0).float()
        pv = self.act(outpvmem)
        if not self.spiking:
            raise Exception('Refractory not allowed in non-spiking mode')
        arp -= output*self.wrp
        self.state = self.NeuronState(
            eps0=eps0.detach(),
            eps1=eps1.detach(),
            arp=arp.detach())

        return output, pv, outpvmem


class QuantConv2dDCLLlayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 im_dims=(28, 28),
                 target_size=10,
                 pooling=None,
                 stride=1,
                 dilation=1,
                 padding=2,
                 alpha=.95,
                 alphas=.9,
                 alpharp=.65,
                 wrp=0,
                 act=nn.Sigmoid(),
                 lc_dropout=False,
                 lc_ampl=.5,
                 spiking=True,
                 random_tau=False,
                 output_layer=False):

        super(QuantConv2dDCLLlayer, self).__init__()
        self.im_dims = im_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lc_ampl = lc_ampl
        self.output_layer = output_layer

        # The following code builds the pooling into the module
        if pooling is not None:
            if not hasattr(pooling, '__len__'):
                pooling = (pooling, pooling)
            pool_pad = ((pooling[0] - 1) // 2,
                        (pooling[1] - 1) // 2)
            self.pooling = pooling
            self.pool = nn.MaxPool2d(
                kernel_size=pooling, stride=pooling, padding=pool_pad)
        else:
            self.pooling = (1, 1)
            self.pool = lambda x: x
        self.kernel_size = kernel_size
        self.target_size = target_size
        if wrp > 0:
            if not spiking:
                raise Exception(
                    'Non-spiking not allowed with refractory neurons')
            self.i2h = QuantContinuousRelativeRefractoryConv2D(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation,
                                                          stride=stride, alpha=alpha, alphas=alphas, alpharp=alpharp, wrp=wrp, act=act, random_tau=random_tau)
        else:
            self.i2h = QuantContinuousConv2D(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation,
                                        stride=stride, alpha=alpha, alphas=alphas, act=act, spiking=spiking, random_tau=random_tau)
        conv_shape = self.i2h.get_output_shape(self.im_dims)
        print('Quant Conv2D Layer ', self.im_dims, conv_shape, self.in_channels,
              self.out_channels, kernel_size, dilation, padding, stride)
        self.output_shape = self.pool(torch.zeros(1, *conv_shape)).shape[1:]
        self.i2o = nn.Linear(np.prod(self.get_flat_size()),
                             target_size, bias=True)
        self.i2o.weight.requires_grad = False
        if lc_dropout is not False:
            self.dropout = torch.nn.Dropout(p=lc_dropout)
        else:
            self.dropout = lambda x: x
        self.i2o.bias.requires_grad = False

        if output_layer:
            self.output_ = nn.Linear(
                np.prod(self.get_flat_size()), target_size, bias=True)

        self.reset_lc_parameters()

    def reset_lc_parameters(self):
        stdv = self.lc_ampl / math.sqrt(self.i2o.weight.size(1))
        self.i2o.weight.data.uniform_(-stdv, stdv)
        if self.i2o.bias is not None:
            self.i2o.bias.data.uniform_(-stdv, stdv)

    def get_flat_size(self):
        w, h = self.get_output_shape()
        return int(w*h*self.out_channels)

    def get_output_shape(self):
        conv_shape = self.i2h.get_output_shape(self.im_dims)
        height = conv_shape[0] // self.pooling[0]
        weight = conv_shape[1] // self.pooling[1]
        return height, weight

    def forward(self, input):
        output, pv, pvmem = self.i2h(input)
        output, pv = self.pool(output), self.pool(pv)
        flatten = pv.view(pv.shape[0], -1)
        pvoutput = self.dropout(self.i2o(flatten))

        if self.output_layer:
            custom_output = self.output_(flatten.detach())
        else:
            custom_output = output

        return custom_output, pvoutput, pv, pvmem

    def init_hiddens(self, batch_size, init_value=0):
        self.i2h.init_state(batch_size, self.im_dims, init_value=init_value)
        return self


