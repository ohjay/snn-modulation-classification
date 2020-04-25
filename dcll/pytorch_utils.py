import torch
import numpy as np
import torchvision.utils as vutils


def tonumpy(x):
    return x.detach().cpu().numpy()


def grad_parameters(module):
    return list(filter(lambda p: p.requires_grad, module.parameters()))


def named_grad_parameters(module):
    return list(filter(lambda p: p[1].requires_grad, module.named_parameters()))


class ForwardHook(object):
    def __init__(self, writer, title, initial_time, debounce_img=150):
        self.writer = writer
        self.title = title
        self.recording_time = initial_time
        self.debounce = debounce_img

    def write_data(self, data, comment=''):
        # if dictionary of 1 item, we use the key as comment
        if isinstance(data, dict) and len(data) == 1:
            comment = '_' + str(list(data)[0])
            data = list(data.values())[0]

        # write the data wrt datatype
        if isinstance(data, dict):
            self.writer.add_scalars(
                self.title + comment, data, self.recording_time)
        # single value
        elif isinstance(data, torch.Tensor) and len(data.shape) == 1 and data.shape[0] == 1:
            self.writer.add_scalar(self.title + comment,
                                   data, self.recording_time)
        elif isinstance(data, torch.Tensor):
            if self.recording_time % self.debounce == 0:
                img = vutils.make_grid(data.unsqueeze(
                    dim=1) / torch.max(data))  # grey color channel
                self.writer.add_image(
                    self.title + comment, img, self.recording_time)
        else:
            raise NotImplementedError

    def __call__(self, ctx, input, output):
        layer_debug = output[-1]
        if not isinstance(layer_debug, list):
            self.write_data(layer_debug)
        else:
            [self.write_data(d, comment='_idx_'+str(i))
             for i, d in enumerate(layer_debug)]
        self.recording_time += 1


class NetworkDumper(object):
    def __init__(self, writer, model):
        self.writer = writer
        self.model = model

    def histogram(self, prefix='', t=0):
        params = self.model.named_parameters()
        for name, param in params:
            self.writer.add_histogram(prefix+name,
                                      param.cpu().detach().numpy().flatten(),
                                      t)

    def weight2d(self, prefix='', t=0):
        params = self.model.named_parameters()
        # filter out all params that don't correspond to convolutions (KCHW)
        params = list(filter(lambda p: len(p[1].shape) == 4 and
                             ((p[1].shape[1] == 1) or
                              (p[1].shape[1] == 2) or
                              (p[1].shape[1] == 3)), params))

        def enforce_1_or_3_channels(p):
            if not p[1].shape[1] == 2:
                return p
            else:
                pad = torch.zeros(p[1].shape[0], 1,
                                  p[1].shape[2], p[1].shape[3])
                new_p = (p[0],
                         torch.cat((p[1].cpu(), pad), dim=1))
                return new_p

        params = list(map(enforce_1_or_3_channels, params))
        all_filters = [
            vutils.make_grid(l[1]).unsqueeze(dim=1)
            for l in params
        ]
        for i, img in enumerate(all_filters):
            self.writer.add_image(prefix+params[i][0],
                                  img, t)

    # cache the current parameters of the model
    def cache(self):
        self.cached = named_grad_parameters(self.model)

    # plot the delta histogram between current and cached parameters (weight updates)
    def diff_histogram(self, prefix='', t=0):
        params = named_grad_parameters(self.model)
        for i, (name, param) in enumerate(params):
            self.writer.add_histogram(prefix+'delta_'+name,
                                      param.cpu().detach().numpy().flatten() -
                                      self.cached[i],
                                      t)

    def start_recording(self, title='forward_data', t=0):
        hook = ForwardHook(self.writer, title, t)
        # register and return the handle
        return self.model.register_forward_hook(hook)
