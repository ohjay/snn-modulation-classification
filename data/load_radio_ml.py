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

    def __init__(self, data_dir, train, normalize=True, per_class_size=10000, val_step=500):
        self.train = train

        if not os.path.exists(os.path.join(data_dir, 'class23.hdf5')):
            # Split huge HDF5 file into per-class HDF5 files

            data_path = os.path.join(data_dir, 'GOLD_XYZ_OSC.0001_1024.hdf5')
            full_h5f = h5py.File(data_path, 'r')

            # Contents of HDF5 file:
            # 'X': (2555904, 1024, 2) float32 array [represents signal]
            # 'Y': (2555904, 24) int64 array [represents class label]
            # 'Z': (2555904, 1) int64 array [represents SNR]

            # Important note:
            # The data is ordered by class, i.e.
            # the first ~1/24th consists of examples of the first class,
            # the second ~1/24th consists of examples of the second class, and so on.

            # Convert one-hot labels back to argmax
            Y = np.argmax(full_h5f['Y'], axis=1)
            for class_idx in range(24):
                class_h5f_path = os.path.join(data_dir, 'class%d.hdf5' % class_idx)
                class_h5f = h5py.File(class_h5f_path, 'w')
                class_h5f.create_dataset('X', data=full_h5f['X'][Y == class_idx, :, :])
                class_h5f.create_dataset('Y', data=Y[Y == class_idx])
                class_h5f.close()
                print('Wrote class %d data to `%s`.' % (class_idx, class_h5f_path))
            Y = None
            full_h5f.close()

        self.X = []
        self.Y = []
        for class_idx in range(24):
            class_h5f_path = os.path.join(data_dir, 'class%d.hdf5' % class_idx)
            class_h5f = h5py.File(class_h5f_path, 'r')
            self.X.append(class_h5f['X'][:per_class_size])
            self.Y.append(class_h5f['Y'][:per_class_size])
            class_h5f.close()
        self.X = np.concatenate(self.X, axis=0)
        self.Y = np.concatenate(self.Y, axis=0)

        # Add fake height dim (TODO remove if switching to 1D convolutions)
        self.X = self.X.transpose(0, 2, 1)[:, :, np.newaxis, :]

        if normalize:
            minval = self.X.min()
            self.X = (self.X - minval) / (self.X.max() - minval)

        # Assign every (val_step)th signal/label to the validation set
        if train:
            self.X = self.X[np.arange(len(self.X)) % val_step != 0]
            self.Y = self.Y[np.arange(len(self.Y)) % val_step != 0]
        else:
            self.X = self.X[::val_step]
            self.Y = self.Y[::val_step]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def get_radio_ml_loader(batch_size, train, taskid=0, **kwargs):
    data_dir = kwargs['data_dir']
    dataset = RadioMLDataset(data_dir, train, normalize=True)
    print('dataset size: %d' % len(dataset))

    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=train)
    loader.taskid = taskid
    loader.name = 'RadioML_{}'.format(taskid)
    loader.short_name = 'RadioML'

    return loader
