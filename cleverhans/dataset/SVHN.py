import numpy as np
import scipy.io

from cleverhans import utils
from cleverhans.dataset.dataset_toolkit import Dataset, convert_image


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