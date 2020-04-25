import torch
import numpy as np


def to_one_hot(t, width):
    t_onehot = torch.zeros(*t.shape+(width,))
    return t_onehot.scatter_(1, t.unsqueeze(-1), 1)


def image2spiketrain(x, y, input_shape, target_size, gain=50, min_duration=None, max_duration=500):
    if min_duration is None:
        min_duration = max_duration - 1

    batch_size = x.shape[0]
    Nin = np.prod(input_shape)
    rates = gain * x.reshape(batch_size, -1)
    p = (1000.0 - np.array(rates)) / 1000  # (batch_size, Nin)
    T = np.random.randint(min_duration, max_duration, batch_size)

    # Generate spike trains
    all_inputs = np.zeros((max_duration, batch_size, Nin))
    for i in range(batch_size):
        spikes = np.ones((T[i], Nin))
        spikes[(np.random.uniform(size=(T[i], Nin)) < p[i]).astype('bool')] = 0
        all_inputs[:T[i], i, :] = spikes

    # The shape of `all_inputs` is (max_duration, batch_size, c, h, w)
    all_inputs = all_inputs.reshape(max_duration, batch_size, *input_shape)

    # The shape of `all_target` is (max_duration, batch_size, target_size)
    all_target = np.repeat(y[np.newaxis, :, :], max_duration, axis=0)

    return all_inputs, all_target
