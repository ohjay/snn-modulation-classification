import torch
import numpy as np


def to_one_hot(t, width):
    t_onehot = torch.zeros(*t.shape+(width,))
    return t_onehot.scatter_(1, t.unsqueeze(-1), 1)


def image2spiketrain(x, y, input_shape,
                     gain=50, min_duration=None, max_duration=500):
    """Convert an image to a frozen Poisson spike train."""
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

def iq2spiketrain(x, y, out_w=28, out_h=28,
                  min_I=-1, max_I=1, min_Q=-1, max_Q=1, max_duration=500):
    """Convert each I/Q sample to a spike in the I/Q plane over time.
    Assumption: the (squeezed) shape of X is (batch, 2, num_timesteps).
    """
    x = x.squeeze()

    # Generate spike trains
    batch_size = x.shape[0]
    num_timesteps = x.shape[-1]
    spike_trains = np.zeros((max_duration, batch_size, 1, out_h, out_w))
    for t in range(min(max_duration, num_timesteps)):
        # Obtain I/Q values
        I_value = x[:, 0, t]
        Q_value = x[:, 1, t]
        # Quantize to cells in image
        cell_I = (I_value - min_I) / (max_I - min_I) * out_w
        cell_Q = (Q_value - min_Q) / (max_Q - min_Q) * out_h
        cell_I = torch.clamp(cell_I, 0, out_w - 1).int()
        cell_Q = torch.clamp(cell_Q, 0, out_h - 1).int()
        # Assign events to samples
        for b in range(batch_size):
            spike_trains[t, b, 0, cell_Q[b], cell_I[b]] = 1

    # The shape of `all_target` is (max_duration, batch_size, target_size)
    all_target = np.repeat(y[np.newaxis, :, :], max_duration, axis=0)

    return spike_trains, all_target
