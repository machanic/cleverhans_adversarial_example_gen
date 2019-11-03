import os

import cv2
import numpy as np

from cleverhans import utils
from cleverhans.dataset.dataset_toolkit import Dataset, convert_image
from cleverhans.generate_adv_script.config import IMAGENET_ALL_FOLDER, ILSVRC12_ROOT_DIR


class MiniImageNet(Dataset):
    NB_CLASSES = 1000
    # NOTE: ImageNet is too big, it doesn't need to train, we load the pretrained model
    def __init__(self, data_dir, train_start=0, train_end=60000, test_start=0,
                 test_end=10000, center=False, max_val=1., num_classes=20,  arch=""):
        kwargs = locals()
        if '__class__' in kwargs:
            del kwargs['__class__']
        super(MiniImageNet, self).__init__(kwargs)
        adv_root_dir = ILSVRC12_ROOT_DIR + "/adversarial_images/{}".format(arch)
        train_clean_file = adv_root_dir + "/clean_untargeted_train.npz"
        test_clean_file = adv_root_dir + "/clean_untargeted_test.npz"
        if os.path.exists(train_clean_file) and os.path.exists(test_clean_file):  # for saving memory, directly load pre-generated data (pixel value range: [0,1])
            train_npz = np.load(train_clean_file)
            test_npz = np.load(test_clean_file)
            x_train, y_train = train_npz["adv_images"], utils.to_categorical(train_npz["gt_label"],
                                                                             nb_classes=num_classes).astype('float32')
            print('npz train data read over')
            x_test, y_test = test_npz["adv_images"], utils.to_categorical(test_npz["gt_label"],
                                                                          nb_classes=num_classes).astype('float32')
            print('npz test data read over')
        else:
            x_test, y_test = data_imagenet_validation(data_dir+"/test", num_classes, train_start=train_start,train_end=train_end,
                                                      test_start=test_start, test_end=test_end)
            print('test original image data read over')
            x_train, y_train = data_imagenet_train(data_dir +"/train", num_classes)  # dummpy useless
            print('train original image data read over')
        self.x_test = x_test
        self.y_test = y_test
        self.x_train = x_train
        self.y_train = y_train

    def to_tensorflow(self, shuffle=4096):
        return (self.in_memory_dataset(self.x_train, self.y_train, shuffle),
                self.in_memory_dataset(self.x_test, self.y_test, repeat=False))


def data_imagenet_validation(datadir, class_num, train_start=0, train_end=60000, test_start=0, test_end=10000):
    assert isinstance(train_start, int)
    assert isinstance(train_end, int)
    assert isinstance(test_start, int)
    assert isinstance(test_end, int)

    x_val = []
    y_val = []
    MiniImageNet_All_Category = sorted(os.listdir(datadir))
    for folder in os.listdir(datadir):
        label = MiniImageNet_All_Category.index(folder)
        for image_file_name in os.listdir(datadir + "/" + folder):
            image_file_path = os.path.join(datadir, folder, image_file_name)
            image = cv2.imread(image_file_path)
            image = cv2.resize(image, (224,224))
            x_val.append(image)
            y_val.append(label)
    x_val = np.asarray(x_val)
    y_val = np.asarray(y_val, dtype=np.int32)
    x_val = convert_image(x_val).astype('float32')
    y_val = utils.to_categorical(y_val, nb_classes=class_num).astype('float32')
    return x_val, y_val


def data_imagenet_train(datadir, class_num, train_start=0, train_end=60000, test_start=0, test_end=10000):
    assert isinstance(train_start, int)
    assert isinstance(train_end, int)
    assert isinstance(test_start, int)
    assert isinstance(test_end, int)
    x_train = []
    y_train = []
    MiniImageNet_All_Category = sorted(os.listdir(datadir))
    for folder in os.listdir(datadir):
        label = MiniImageNet_All_Category.index(folder)
        for image_file_name in os.listdir(datadir + "/" + folder):
            image_file_path = os.path.join(datadir, folder, image_file_name)
            image = cv2.imread(image_file_path)  # FIXME BGR
            image = cv2.resize(image, (224,224))
            x_train.append(image)
            y_train.append(label)
    x_train= np.asarray(x_train)
    y_train = np.asarray(y_train, dtype=np.int32)
    x_train = convert_image(x_train).astype('float32')
    y_train = utils.to_categorical(y_train, nb_classes=class_num).astype('float32')
    return x_train,y_train