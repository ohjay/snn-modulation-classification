import os
import h5py
import torch
import numpy as np
from torch.utils import data
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader


class RadioMLDataset(data.Dataset):
    """RadioML data.
    Available here: https://www.deepsig.io/datasets.
    """

    def __init__(self, data_dir, train, normalize=True, dataset_size=200000):
        self.train = train
        data_path = os.path.join(data_dir, 'GOLD_XYZ_OSC.0001_1024.hdf5')

        # Contents of HDF5 file:
        # 'X': (2555904, 1024, 2) float32 array [represents signal]
        # 'Y': (2555904, 24) int64 array [represents class label]
        # 'Z': (2555904, 1) int64 array [represents SNR]

        h5f = h5py.File(data_path, 'r')
        self.X = h5f['X'][:dataset_size]
        self.Y = h5f['Y'][:dataset_size]
        h5f.close()

        # Add fake channel dim
        self.X = self.X[:, np.newaxis, ...]

        # Convert one-hot labels back to argmax
        self.Y = np.argmax(self.Y, axis=1)

        if normalize:
            minval = self.X.min()
            self.X = (self.X - minval) / (self.X.max() - minval)

        # Define splits according to a ratio of 3686 train : 410 validation
        # Every 10th signal/label/SNR should be assigned to the validation set
        if train:
            self.X = self.X[np.arange(len(self.X)) % 10 != 0]
            self.Y = self.Y[np.arange(len(self.Y)) % 10 != 0]
        else:
            self.X = self.X[::10]
            self.Y = self.Y[::10]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def get_radio_ml_loader(batch_size, train, taskid=0, **loader_kwargs):
    data_dir = '/media/owen/ba9d40b5-89de-4832-bad4-156b118e4a66/2018.01'
    dataset = RadioMLDataset(data_dir, train, normalize=True)
    print('dataset size: %d' % len(dataset))

    loader = DataLoader(dataset=dataset, batch_size=batch_size,
                        shuffle=train, **loader_kwargs)
    loader.taskid = taskid
    loader.name = 'RadioML_{}'.format(taskid)
    loader.short_name = 'RadioML'

    return loader
