import os

import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.keras.datasets.cifar100 import load_data as load_cifar100_data


from cleverhans.dataset.dataset_toolkit import Dataset, convert_image, np_utils


class CIFAR100(Dataset):
    """The CIFAR-10 dataset"""

    NB_CLASSES = 100

    def __init__(self, data_dir, dataset_name, train_start=0, train_end=60000, test_start=0,
                 test_end=10000, center=False, max_val=1.):
        kwargs = locals()
        if '__class__' in kwargs:
            del kwargs['__class__']
        super(CIFAR100, self).__init__(kwargs)
        is_fine_label = True if dataset_name == "CIFAR100" else False
        packed = data_cifar100(data_dir, is_fine_label, train_start=train_start,
                              train_end=train_end,
                              test_start=test_start,
                              test_end=test_end)
        x_train, y_train, x_test, y_test = packed

        if center:
            x_train = x_train * 2. - 1.
            x_test = x_test * 2. - 1.
        x_train *= max_val
        x_test *= max_val

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.max_val = max_val

    def to_tensorflow(self, shuffle=4096):
        # This is much more efficient with data augmentation, see tutorials.
        return (self.in_memory_dataset(self.x_train, self.y_train, shuffle),
                self.in_memory_dataset(self.x_test, self.y_test, repeat=False))


class CIFAR10(Dataset):
    """The CIFAR-10 dataset"""

    NB_CLASSES = 10

    LABEL_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog",
                   "frog", "horse", "ship", "truck"]

    def __init__(self, data_dir, train_start=0, train_end=60000, test_start=0,
                 test_end=10000, center=False, max_val=1.):
        kwargs = locals()
        if '__class__' in kwargs:
            del kwargs['__class__']
        super(CIFAR10, self).__init__(kwargs)
        packed = data_cifar10(data_dir, train_start=train_start,
                              train_end=train_end,
                              test_start=test_start,
                              test_end=test_end)
        x_train, y_train, x_test, y_test = packed

        if center:
            x_train = x_train * 2. - 1.
            x_test = x_test * 2. - 1.
        x_train *= max_val
        x_test *= max_val

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.max_val = max_val

    def to_tensorflow(self, shuffle=4096):
        # This is much more efficient with data augmentation, see tutorials.
        return (self.in_memory_dataset(self.x_train, self.y_train, shuffle),
                self.in_memory_dataset(self.x_test, self.y_test, repeat=False))


def data_cifar10(data_dir, train_start=0, train_end=50000, test_start=0, test_end=10000):
    """
    一次性返回所有数据，load_batch全部读取出来5个文件
    Preprocess CIFAR10 dataset
    :return:
    """

    # These values are specific to CIFAR10
    def load_data(dirname):

        """Loads CIFAR10 dataset.
        Returns:
            Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
        """

        num_train_samples = 50000

        x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
        y_train = np.empty((num_train_samples,), dtype='uint8')

        for i in range(1, 6):
            fpath = os.path.join(dirname, 'data_batch_' + str(i))
            (x_train[(i - 1) * 10000:i * 10000, :, :, :],
             y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

        fpath = os.path.join(dirname, 'test_batch')
        x_test, y_test = load_batch(fpath)

        y_train = np.reshape(y_train, (len(y_train), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))

        if K.image_data_format() == 'channels_last':
            x_train = x_train.transpose(0, 2, 3, 1)
            x_test = x_test.transpose(0, 2, 3, 1)

        return (x_train, y_train), (x_test, y_test)

    img_rows = 32
    img_cols = 32
    nb_classes = 10

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = load_data(data_dir)

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)

    x_train = convert_image(x_train)
    x_test = convert_image(x_test)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    x_train = x_train[train_start:train_end, :, :, :]
    y_train = y_train[train_start:train_end, :]
    x_test = x_test[test_start:test_end, :]
    y_test = y_test[test_start:test_end, :]

    return x_train, y_train, x_test, y_test

def data_cifar100(data_dir, is_fine_label, train_start=0, train_end=50000, test_start=0, test_end=10000):
    """
    一次性返回所有数据，load_batch全部读取出来5个文件
    Preprocess CIFAR10 dataset
    :return:
    """

    # These values are specific to CIFAR10
    def load_data(dirname):
        """Loads CIFAR10 dataset.
        Returns:
            Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
        """
        # num_train_samples = 50000
        # x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
        # y_train = np.empty((num_train_samples,), dtype='uint8')
        train_path = os.path.join(dirname, 'train')
        test_path = os.path.join(dirname, 'test')
        label_key = "fine_labels" if is_fine_label else "coarse_labels"
        x_train, y_train = load_batch(train_path,label_key=label_key)
        x_test, y_test = load_batch(test_path,label_key=label_key)

        y_train = np.reshape(y_train, (len(y_train), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))

        if K.image_data_format() == 'channels_last':
            x_train = x_train.transpose(0, 2, 3, 1)
            x_test = x_test.transpose(0, 2, 3, 1)

        return (x_train, y_train), (x_test, y_test)

    img_rows = 32
    img_cols = 32
    if is_fine_label:
        nb_classes = 100
    else:
        nb_classes = 20
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = load_data(data_dir)

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)

    x_train = convert_image(x_train)
    x_test = convert_image(x_test)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    x_train = x_train[train_start:train_end, :, :, :]
    y_train = y_train[train_start:train_end, :]
    x_test = x_test[test_start:test_end, :]
    y_test = y_test[test_start:test_end, :]

    return x_train, y_train, x_test, y_test