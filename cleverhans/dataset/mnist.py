import array
import functools
import gzip
import operator
import os
import struct
import sys
import tempfile

import numpy as np

from cleverhans import utils
from cleverhans.dataset.dataset_toolkit import Dataset, convert_image, maybe_download_file


class MNIST(Dataset):
    """The MNIST dataset"""

    NB_CLASSES = 10

    def __init__(self, data_dir, train_start=0, train_end=60000, test_start=0,
                 test_end=10000, center=False, max_val=1.):
        kwargs = locals()
        if '__class__' in kwargs:
            del kwargs['__class__']
        super(MNIST, self).__init__(kwargs)
        x_train, y_train, x_test, y_test = data_mnist(data_dir, train_start=train_start,
                                                      train_end=train_end,
                                                      test_start=test_start,
                                                      test_end=test_end)

        if center:
            x_train = x_train * 2. - 1.
            x_test = x_test * 2. - 1.
        x_train *= max_val
        x_test *= max_val

        self.x_train = x_train.astype('float32')
        self.y_train = y_train.astype('float32')
        self.x_test = x_test.astype('float32')
        self.y_test = y_test.astype('float32')

    def to_tensorflow(self, shuffle=4096):
        return (self.in_memory_dataset(self.x_train, self.y_train, shuffle),
                self.in_memory_dataset(self.x_test, self.y_test, repeat=False))


def data_mnist(datadir=tempfile.gettempdir(), train_start=0,
               train_end=60000, test_start=0, test_end=10000):
    """
    Load and preprocess MNIST dataset
    :param datadir: path to folder where data should be stored
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :return: tuple of four arrays containing training data, training labels,
             testing data and testing labels.
    """
    assert isinstance(train_start, int)
    assert isinstance(train_end, int)
    assert isinstance(test_start, int)
    assert isinstance(test_end, int)

    X_train = download_and_parse_mnist_file(
        'train-images-idx3-ubyte.gz', datadir=datadir)
    Y_train = download_and_parse_mnist_file(
        'train-labels-idx1-ubyte.gz', datadir=datadir)
    X_test = download_and_parse_mnist_file(
        't10k-images-idx3-ubyte.gz', datadir=datadir)
    Y_test = download_and_parse_mnist_file(
        't10k-labels-idx1-ubyte.gz', datadir=datadir)

    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    X_train = convert_image(X_train)
    X_test = convert_image(X_test)
    X_train = X_train[train_start:train_end]
    Y_train = Y_train[train_start:train_end]
    X_test = X_test[test_start:test_end]
    Y_test = Y_test[test_start:test_end]

    Y_train = utils.to_categorical(Y_train, nb_classes=10)
    Y_test = utils.to_categorical(Y_test, nb_classes=10)
    return X_train, Y_train, X_test, Y_test


def download_and_parse_mnist_file(file_name, datadir=None, force=False):
    url = os.path.join('http://yann.lecun.com/exdb/mnist/', file_name)
    file_name = maybe_download_file(url, datadir=datadir, force=force)

    # Open the file and unzip it if necessary
    if os.path.splitext(file_name)[1] == '.gz':
        open_fn = gzip.open
    else:
        open_fn = open

    # Parse the file
    with open_fn(file_name, 'rb') as file_descriptor:
        header = file_descriptor.read(4)
        assert len(header) == 4

        zeros, data_type, n_dims = struct.unpack('>HBB', header)
        assert zeros == 0

        hex_to_data_type = {
            0x08: 'B',
            0x09: 'b',
            0x0b: 'h',
            0x0c: 'i',
            0x0d: 'f',
            0x0e: 'd'}
        data_type = hex_to_data_type[data_type]

        # data_type unicode to ascii conversion (Python2 fix)
        if sys.version_info[0] < 3:
            data_type = data_type.encode('ascii', 'ignore')

        dim_sizes = struct.unpack(
            '>' + 'I' * n_dims,
            file_descriptor.read(4 * n_dims))

        data = array.array(data_type, file_descriptor.read())
        data.byteswap()

        desired_items = functools.reduce(operator.mul, dim_sizes)
        assert len(data) == desired_items
        return np.array(data).reshape(dim_sizes)