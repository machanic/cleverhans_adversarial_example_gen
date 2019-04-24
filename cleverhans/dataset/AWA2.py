import numpy as np
import scipy.io

from cleverhans import utils
from cleverhans.dataset.dataset_toolkit import Dataset, convert_image
from cleverhans.generate_adv_script.config import AWA2_ALL_FOLDER,AWA2_TRAIN_CLASSES,AWA2_TEST_CLASSES,IMG_SIZE,AWA2_SOURCE_DATA_DIR
import os
import cv2

class AWA2(Dataset):
    """The SVHN dataset"""

    NB_CLASSES = len(AWA2_ALL_FOLDER)

    def __init__(self, data_dir):
        kwargs = locals()
        if '__class__' in kwargs:
            del kwargs['__class__']
        super(AWA2, self).__init__(kwargs)
        self.data_dir = data_dir
        self.x_train, self.y_train, self.x_test, self.y_test = data_AWA2(AWA2_SOURCE_DATA_DIR + "/Animals_with_Attributes2/JPEGImages/")

    def to_tensorflow(self, shuffle=4096):
        return (self.in_memory_dataset(self.x_train, self.y_train, shuffle),
                self.in_memory_dataset(self.x_test, self.y_test, repeat=False))

def data_AWA2(datadir, class_num=len(AWA2_ALL_FOLDER)):

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for folder in os.listdir(datadir):

        label = AWA2_ALL_FOLDER.index(folder)
        if folder in AWA2_TRAIN_CLASSES:
            x = x_train
            y = y_train
        elif folder in AWA2_TEST_CLASSES:
            x = x_test
            y = y_test
        else:
            raise Exception("class {} is not in train and test".format(folder))
        for image_file_name in os.listdir(datadir + "/" + folder):
            image_file_path = os.path.join(datadir, folder, image_file_name)
            image = cv2.imread(image_file_path)
            image = cv2.resize(image, (IMG_SIZE["AWA2"], IMG_SIZE["AWA2"]))
            x.append(image)
            y.append(label)

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)
    x_train = convert_image(x_train)
    x_test = convert_image(x_test)
    y_train = utils.to_categorical(np.array(y_train), nb_classes=class_num).astype(np.float32)
    y_test = utils.to_categorical(np.array(y_test),nb_classes=class_num).astype(np.float32)

    return x_train, y_train, x_test, y_test