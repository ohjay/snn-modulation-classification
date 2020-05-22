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

    def __init__(self, data_dir, train,
                 normalize=True, min_snr=6, per_h5_frac=0.2, train_frac=0.9):

        self.train = train

        if train:
            print("Load training set")
        else:
            print("Load test set")

        if os.path.exists('test_y.npy'):
            print("Start loading from npy file")
            if train:
                self.X = np.load('train_x.npy')
                self.Y = np.load('train_y.npy')
            else:
                self.X = np.load('test_x.npy')
                self.Y = np.load('test_y.npy')
            print("Data loaded from npy file")
            return


        if not os.path.exists(os.path.join(data_dir, 'class23_snr30.hdf5')):
            # Split huge HDF5 file into per-SNR, per-class HDF5 files

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

            # There are 24 different modulation classes.
            # For each class, there are 2555904 / 24 = 106496 examples.
            # There are 26 different SNR levels: -20, -18, -16, ..., 28, 30.
            # For each (class, SNR) pair, there are 106496 / 26 = 4096 examples.

            # Convert one-hot labels back to argmax
            Y = np.argmax(full_h5f['Y'], axis=1)

            for class_idx in range(24):
                class_X = full_h5f['X'][Y == class_idx, :, :]
                class_Z = full_h5f['Z'][Y == class_idx, 0]
                for snr in range(-26, 32, 2):
                    class_snr_name = 'class%d_snr%d.hdf5' % (class_idx, snr)
                    h5f_path = os.path.join(data_dir, class_snr_name)
                    h5f = h5py.File(h5f_path, 'w')
                    h5f.create_dataset('X', data=class_X[class_Z == snr, :, :])
                    h5f.close()
                    print('Wrote (SNR {z}, class {cl}) data to `{path}`.'.format(
                        z=snr, cl=class_idx, path=h5f_path))
                class_X = None
                class_Z = None
            Y = None
            full_h5f.close()

        # Min/max values across entire dataset
        # Want to use the same values for train/test normalization
        X_minval = float('inf')
        X_maxval = float('-inf')

        # The data for each (class, SNR) pair
        # will be truncated to the first PER_H5_SIZE examples
        per_h5_size = int(per_h5_frac * 4096)
        snr_count = (30 - min_snr) // 2 + 1
        train_split_size = int(train_frac * per_h5_size)
        if train:
            split_size = train_split_size
        else:
            split_size = per_h5_size - train_split_size
        total_size = 24 * snr_count * split_size

        self.X = np.zeros((total_size, 1024, 2), dtype=np.float32)
        self.Y = np.zeros(total_size, dtype=np.int64)
        for class_idx in range(1):
            print("Load class {}".format(class_idx))
            for snr_idx, snr in enumerate(range(min_snr, 32, 2)):
                class_snr_name = 'class%d_snr%d.hdf5' % (class_idx, snr)
                h5f_path = os.path.join(data_dir, class_snr_name)
                h5f = h5py.File(h5f_path, 'r')
                X = h5f['X'][:]
                X_minval = min(X_minval, X.min())
                X_maxval = max(X_maxval, X.max())
                if train:
                    X_split = X[:train_split_size]
                else:
                    X_split = X[train_split_size:per_h5_size]
                # Interleave
                start_idx = (class_idx * snr_count) + snr_idx
                self.X[start_idx::24*snr_count] = X_split
                self.Y[start_idx::24*snr_count] = class_idx
                h5f.close()
                X = None
                X_split = None

        # Add fake height dim (TODO remove if switching to 1D convolutions)
        self.X = self.X.transpose(0, 2, 1)[:, :, np.newaxis, :]

        if normalize:
            self.X = (self.X - X_minval) / (X_maxval - X_minval)

        print("Save to npy file")
        if train:
            np.save('train_x.npy', self.X)
            np.save('train_y.npy', self.Y)
        else:
            np.save('test_x.npy', self.X)
            np.save('test_y.npy', self.Y)

        print("Saved npy")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def get_radio_ml_loader(batch_size, train, taskid=0, **kwargs):
    data_dir = kwargs['data_dir']
    dataset = RadioMLDataset(data_dir, train, normalize=False)
    print('dataset size: %d' % len(dataset))

    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=train)
    loader.taskid = taskid
    loader.name = 'RadioML_{}'.format(taskid)
    loader.short_name = 'RadioML'

    return loader
