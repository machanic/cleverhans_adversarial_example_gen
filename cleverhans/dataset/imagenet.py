import os

import cv2
import numpy as np

from cleverhans import utils
from cleverhans.dataset.dataset_toolkit import Dataset
from cleverhans.generate_adv_script.config import IMAGENET_ALL_FOLDER


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
    y_train = np.asarray(y_train, dtype=np.int32)
    y_train = utils.to_categorical(y_train, nb_classes=class_num)
    return np.asarray(x_train, dtype=np.float32),y_train