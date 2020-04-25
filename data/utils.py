import torch
import numpy as np


def to_one_hot(t, width):
    t_onehot = torch.zeros(*t.shape+(width,))
    return t_onehot.scatter_(1, t.unsqueeze(-1), 1)


def __gen_ST(N, T, rate, mode='regular'):
    if mode == 'regular':
        spikes = np.zeros([T, N])
        spikes[::(1000//rate)] = 1
        return spikes
    elif mode == 'poisson':
        spikes = np.ones([T, N])
        spikes[np.random.binomial(
            1, float(1000. - rate)/1000, size=(T, N)).astype('bool')] = 0
        return spikes
    else:
        raise Exception('mode must be regular or Poisson')


def spiketrains(N, T, rates, mode='poisson'):
    if not hasattr(rates, '__iter__'):
        return __gen_ST(N, T, rates, mode)
    rates = np.array(rates)
    M = rates.shape[0]
    spikes = np.zeros([T, N])
    for i in range(M):
        if int(rates[i]) > 0:
            spikes[:, i] = __gen_ST(1, T, int(rates[i]), mode=mode).flatten()
    return spikes


def image2spiketrain(x, y, input_shape, target_size, gain=50, min_duration=None, max_duration=500):
    if min_duration is None:
        min_duration = max_duration - 1
    batch_size = x.shape[0]
    T = np.random.randint(min_duration, max_duration, batch_size)
    Nin = np.prod(input_shape)
    allinputs = np.zeros([batch_size, max_duration, Nin])
    for i in range(batch_size):
        st = spiketrains(T=T[i], N=Nin, rates=gain *
                         x[i].reshape(-1)).astype(np.float32)
        allinputs[i] = np.pad(
            st, ((0, max_duration - T[i]), (0, 0)), 'constant')
    allinputs = np.transpose(allinputs, (1, 0, 2))
    allinputs = allinputs.reshape(
        allinputs.shape[0], allinputs.shape[1], *input_shape)

    alltgt = np.zeros(
        [max_duration, batch_size, target_size], dtype=np.float32)
    for i in range(batch_size):
        alltgt[:, i, :] = y[i]

    return allinputs, alltgt
