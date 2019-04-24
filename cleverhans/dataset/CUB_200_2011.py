import numpy as np
import scipy.io

from cleverhans import utils
from cleverhans.dataset.dataset_toolkit import Dataset, convert_image
from cleverhans.generate_adv_script.config import IMG_SIZE,CLASS_NUM
import os
import cv2

class CUB200_2011(Dataset):
    """The SVHN dataset"""

    NB_CLASSES = CLASS_NUM["CUB"]

    def __init__(self, data_dir):
        kwargs = locals()
        if '__class__' in kwargs:
            del kwargs['__class__']
        super(CUB200_2011, self).__init__(kwargs)
        self.data_dir = data_dir
        self.img_id_dict = self.read_img_path_txt(self.data_dir + "/images.txt")
        self.label_id_dict = self.read_img_label_txt(self.data_dir +"/image_class_labels.txt")
        train_path_list, test_path_list = self.read_train_and_test_paths(self.data_dir +"/train_test_split.txt",self.img_id_dict,self.label_id_dict)

        self.x_train, self.y_train, self.x_test, self.y_test = self.read_data(train_path_list, test_path_list, CLASS_NUM["CUB"])

    def to_tensorflow(self, shuffle=4096):
        return (self.in_memory_dataset(self.x_train, self.y_train, shuffle),
                self.in_memory_dataset(self.x_test, self.y_test, repeat=False))

    def read_img_path_txt(self, img_txt_path):
        img_dict = {}
        with open(img_txt_path, "r") as file_obj:
            for line in file_obj:
                img_id, img_path = line.split(" ")
                img_dict[int(img_id)] = self.data_dir + "/images/" + img_path.strip()
        return img_dict

    def read_img_label_txt(self, label_txt_path):
        label_dict = {}
        with open(label_txt_path, "r") as file_obj:
            for line in file_obj:
                img_id, label = line.split()
                label = int(label) - 1
                label_dict[int(img_id)] = label
        return label_dict

    def read_train_and_test_paths(self, train_test_split_path, img_id_dict, label_id_dict):
        train_path_list = []
        test_path_list = []
        with open(train_test_split_path, "r") as file_obj:
            for line in file_obj:
                img_id, is_train = line.strip().split()
                is_train = int(is_train)
                img_path = img_id_dict[int(img_id)]
                label = label_id_dict[int(img_id)]
                if is_train == 1:
                    train_path_list.append((img_path, label))
                else:
                    test_path_list.append((img_path, label))
        return train_path_list, test_path_list

    def read_data(self, train_path_list, test_path_list, class_num):
        x_train = []
        y_train = []
        x_test = []
        y_test = []

        for image_file_path,label in train_path_list:
            image = cv2.imread(image_file_path)
            image = cv2.resize(image, (IMG_SIZE["CUB"], IMG_SIZE["CUB"]))
            x_train.append(image)
            y_train.append(label)

        for image_file_path, label in test_path_list:
            image = cv2.imread(image_file_path)
            image = cv2.resize(image, (IMG_SIZE["CUB"], IMG_SIZE["CUB"]))
            x_test.append(image)
            y_test.append(label)
        x_train = np.asarray(x_train)
        x_test = np.asarray(x_test)
        x_train = convert_image(x_train)
        x_test = convert_image(x_test)
        y_train = utils.to_categorical(np.array(y_train), nb_classes=class_num).astype(np.float32)
        y_test = utils.to_categorical(np.array(y_test), nb_classes=class_num).astype(np.float32)
        return x_train, y_train, x_test, y_test
