"""Dataset class for CleverHans

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tempfile
import warnings
import numpy as np
import tensorflow as tf
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

