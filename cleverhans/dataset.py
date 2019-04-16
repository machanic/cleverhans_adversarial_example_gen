"""Dataset class for CleverHans

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import array
import functools
import gzip
import scipy.io

import operator
import os
import struct
import tempfile
import sys
import warnings
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch
import tensorflow as tf
from cleverhans.generate_adv_script.config import IMAGENET_ALL_FOLDER
import torchvision.transforms as transforms
import copy
from PIL import Image
from collections import defaultdict
import pickle


try:
    from tensorflow.python.keras.utils import np_utils
except ImportError:
    # In tf 1.8, np_utils doesn't seem to be publicly exposed.
    # In later tf versions, it is, and in pre-tf keras it was too.
    from tensorflow.python.keras import _impl

    np_utils = _impl.keras.utils.np_utils
    # In tf 1.8, "from tensorflow.keras.datasets import cifar10" doesn't work even though the module exists
    warnings.warn("Support for TensorFlow versions prior to 1.12 is deprecated."
                  " CleverHans using earlier versions may quit working on or after 2019-07-07.")
from cleverhans import utils


class Dataset(object):
    """Abstract base class representing a dataset.
    """

    # The number of classes in the dataset. Should be specified by subclasses.
    NB_CLASSES = None

    def __init__(self, kwargs=None):
        if kwargs is None:
            kwargs = {}
        if "self" in kwargs:
            del kwargs["self"]
        self.kwargs = kwargs

    def get_factory(self):
        """Returns a picklable callable that recreates the dataset.
        """

        return Factory(type(self), self.kwargs)

    def get_set(self, which_set):
        """Returns the training set or test set as an (x_data, y_data) tuple.
        :param which_set: 'train' or 'test'
        """
        return (getattr(self, 'x_' + which_set),
                getattr(self, 'y_' + which_set))

    def to_tensorflow(self):
        raise NotImplementedError()

    @classmethod
    def in_memory_dataset(cls, x, y, shuffle=None, repeat=True):
        assert x.shape[0] == y.shape[0]
        d = tf.data.Dataset.range(x.shape[0])
        if repeat:
            d = d.repeat()
        if shuffle:
            d = d.shuffle(shuffle)

        def lookup(p):
            return x[p], y[p]

        d = d.map(lambda i: tf.py_func(lookup, [i], [tf.float32] * 2))
        return d


class ImageNet(Dataset):
    NB_CLASSES = 1000
    # NOTE: ImageNet is too big, it doesn't need to train, we load the pretrained model
    def __init__(self, data_dir, train_start=0, train_end=60000, test_start=0,
                 test_end=10000, center=False, max_val=1.):
        kwargs = locals()
        if '__class__' in kwargs:
            del kwargs['__class__']
        super(ImageNet, self).__init__(kwargs)
        x_test, y_test = data_imagenet_validation(data_dir,1000, train_start=train_start,train_end=train_end,
                                                  test_start=test_start, test_end=test_end)
        x_train, y_train = data_imagenet_train(data_dir, 1000)  # dummpy useless
        # if center:
        #     x_test = x_test * 2. - 1.
        # x_test *= max_val

        self.x_test = x_test.astype('float32')
        self.y_test = y_test.astype('float32')
        self.x_train = x_train
        self.y_train = y_train
        # self.x_train = x_train.astype('float32')
        # self.y_train = y_train.astype('float32')


    def to_tensorflow(self, shuffle=4096):
        # This is much more efficient with data augmentation, see tutorials.
        return (self.in_memory_dataset(self.x_test, self.y_test, repeat=False),)

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


class CIFAR100(Dataset):
    """The CIFAR-10 dataset"""

    NB_CLASSES = 100

    def __init__(self, data_dir, train_start=0, train_end=60000, test_start=0,
                 test_end=10000, center=False, max_val=1.):
        kwargs = locals()
        if '__class__' in kwargs:
            del kwargs['__class__']
        super(CIFAR100, self).__init__(kwargs)
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

class SVHN(Dataset):
    """The SVHN dataset"""

    NB_CLASSES = 10

    def __init__(self, data_dir):
        kwargs = locals()
        if '__class__' in kwargs:
            del kwargs['__class__']
        super(SVHN, self).__init__(kwargs)
        self.data_dir = data_dir
        self.train_file_path = data_dir + '/train_32x32.mat'
        self.test_file_path = data_dir + "/test_32x32.mat"
        train_data_and_label = scipy.io.loadmat(self.train_file_path)
        self.x_train = convert_image(np.transpose(train_data_and_label["X"], axes=(3,0,1,2))).astype(np.float32)
        self.y_train = utils.to_categorical(train_data_and_label["y"].reshape(-1, 1) - 1, nb_classes=10).astype(np.float32)
        test_data_and_label = scipy.io.loadmat(self.test_file_path)
        self.x_test = convert_image(np.transpose(test_data_and_label["X"], axes=(3,0,1,2))).astype(np.float32)
        self.y_test = utils.to_categorical(test_data_and_label["y"].reshape(-1, 1) - 1, nb_classes=10).astype(np.float32)

    def to_tensorflow(self, shuffle=4096):
        return (self.in_memory_dataset(self.x_train, self.y_train, shuffle),
                self.in_memory_dataset(self.x_test, self.y_test, repeat=False))


class Factory(object):
    """
    A callable that creates an object of the specified type and configuration.
    """

    def __init__(self, cls, kwargs):
        self.cls = cls
        self.kwargs = kwargs

    def __call__(self):
        """Returns the created object.
        """
        return self.cls(**self.kwargs)


def maybe_download_file(url, datadir=None, force=False):
    try:
        from urllib.request import urlretrieve
    except ImportError:
        from urllib import urlretrieve

    if not datadir:
        datadir = tempfile.gettempdir()
    file_name = url[url.rfind("/") + 1:]
    dest_file = os.path.join(datadir, file_name)

    isfile = os.path.isfile(dest_file)

    if force or not isfile:
        urlretrieve(url, dest_file)
    return dest_file


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


def data_imagenet_validation(datadir, class_num, train_start=0, train_end=60000, test_start=0, test_end=10000):
    assert isinstance(train_start, int)
    assert isinstance(train_end, int)
    assert isinstance(test_start, int)
    assert isinstance(test_end, int)
    num_train_samples = 50000

    x_val = np.empty((num_train_samples, 224, 224, 3), dtype='uint8')
    y_val = np.empty((num_train_samples,), dtype='uint8')

    index = 0
    for folder in os.listdir(datadir):
        label = IMAGENET_ALL_FOLDER.index(folder)
        for image_file_name in os.listdir(datadir + "/" + folder):
            image_file_path = os.path.join(datadir, folder, image_file_name)
            image = cv2.imread(image_file_path)
            image = cv2.resize(image, (224,224))
            x_val[index] = image
            y_val[index] = label
            index += 1
    y_val = utils.to_categorical(y_val, nb_classes=class_num)
    return x_val, y_val

def data_imagenet_train(datadir, class_num, train_start=0, train_end=60000, test_start=0, test_end=10000):
    assert isinstance(train_start, int)
    assert isinstance(train_end, int)
    assert isinstance(test_start, int)
    assert isinstance(test_end, int)
    x_train = []
    y_train = []
    num_train_samples = 50000
    x_val = np.zeros((num_train_samples, 224, 224, 3), dtype='uint8')
    y_val = np.zeros((num_train_samples,), dtype='uint8')
    y_val = utils.to_categorical(y_val, nb_classes=class_num)
    index = 0
    for folder in os.listdir(datadir):
        label = IMAGENET_ALL_FOLDER.index(folder)
        for image_file_name in os.listdir(datadir + "/" + folder):
            image_file_path = os.path.join(datadir, folder, image_file_name)
            image = cv2.imread(image_file_path)  # FIXME BGR
            image = cv2.resize(image, (224,224))
            x_train.append(image)
            y_train.append(label)
            index += 1
    return np.asarray(x_val, dtype=np.float32), np.asarray(y_val, dtype=np.int32)

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


def get_category_similarity(similar_category_path):
    word_sim_candidates = defaultdict(list)
    with open(similar_category_path, "rb") as file_obj:
        pair_score = pickle.load(file_obj)
        for (a, b), score in pair_score.items():
            word_sim_candidates[a].append((b, score))
            word_sim_candidates[b].append((a, score))
    for word, candidates in word_sim_candidates.items():
        word_sim_candidates[word] = sorted(candidates, key=lambda e: e[1], reverse=True)
    return word_sim_candidates


def get_preprocessor():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalizer = transforms.Normalize(mean=mean, std=std)
    preprocess_transform = transforms.Compose([
        transforms.ToTensor(),
        normalizer
    ])
    return preprocess_transform


def transform_image(preprocessor, np_image):
    processed = []
    for image in np_image:
        image_processed = preprocessor(Image.fromarray(image.astype(np.uint8))).detach().numpy()
        processed.append(image_processed)
    result = np.stack(processed)  # B, C, H, W
    result = np.ascontiguousarray(np.transpose(result, axes=(0, 2, 3, 1)), dtype=np.float32)
    return result


def convert_image(np_image):
    np_image = np_image.astype(np.float32)
    np_image = np.divide(np_image, 255.0)
    if np_image.shape[-1] == 1:
        mean = np.array([0.456])[None][None][None]  # 单通道的时候用中间这个数字作为均值
        std = np.array([0.224])[None][None][None]
    elif np_image.shape[-1] == 3:
        mean = np.array([0.485, 0.456, 0.406])[None][None][None]
        std = np.array([0.229, 0.224, 0.225])[None][None][None]
    np_image = (np_image - mean) / std
    return np_image


def recreate_image(im_as_tensor):
    """
        Recreates images from a torch Tensor, sort of reverse preprocessing

    Args:
        im_as_tensor (torch variable): Image to recreate

    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1 / 0.229, 1 / 0.224, 1 / 0.225]
    recreated_im = copy.copy(im_as_tensor.detach().cpu().numpy()[0])  # C, H, W
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)  # H, W, C
    return Image.fromarray(recreated_im)
