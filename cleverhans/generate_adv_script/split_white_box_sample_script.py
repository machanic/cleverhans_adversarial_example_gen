import sys
sys.path.append("/home1/machen/adversarial_example")
from cleverhans.generate_adv_script.config import CIFAR10_OUTPUT_DATA_DIR,CIFAR100_OUTPUT_DATA_DIR,\
                        MNIST_OUTPUT_DATA_DIR, FMNIST_OUTPUT_DATA_DIR, \
                        META_ATTACKER_INDEX,META_ATTACKER_PART_I,META_ATTACKER_PART_II, ROOT_DATA_DIR, SVHN_OUTPUT_DATA_DIR
import os
import numpy as np
from collections import defaultdict
import random
import argparse
import glob
import re


def chunk(xs, n):
    ys = list(xs)
    random.shuffle(ys)
    size = len(ys) // n
    leftovers= ys[size*n:]
    for c in range(n):
        if leftovers:
           extra= [ leftovers.pop() ]
        else:
           extra= []
        yield ys[c*size:(c+1)*size] + extra


def process_data(attack_name, data_json, output_root_dir):
    attack_index = META_ATTACKER_INDEX.index(attack_name) + 1
    adv_images = data_json["adv_images"]
    gt_label = data_json["gt_label"]
    adv_image_dict = defaultdict(list)
    for idx, label in enumerate(gt_label):
        adv_image_dict[label].append(adv_images[idx])
    for label, adv_images_list in adv_image_dict.items():
        adv_images = np.stack(adv_images_list)
        out_dir_name = output_root_dir + "/{}_{}".format(label, attack_index)
        support_dir = "{}/support".format(out_dir_name)
        query_dir = "{}/query".format(out_dir_name)
        os.makedirs(support_dir, exist_ok=True)
        os.makedirs(query_dir, exist_ok=True)
        all_index = np.arange(len(adv_images_list))
        sub_lists = list(chunk(all_index, 2))
        support_indexes = sub_lists[0]
        query_indexes = sub_lists[1]
        support_images = adv_images[support_indexes]
        query_images = adv_images[query_indexes]

        out_file_path = "{}/{}.npy".format(support_dir, "support")
        count_file_path = "{}/count.txt".format(support_dir)
        fp = np.memmap(out_file_path, dtype='float32', mode='w+', shape=support_images.shape)
        fp[:, :, :, :] = support_images[:, :, :, :]
        del fp
        with open(count_file_path, "w") as file_obj:
            file_obj.write(str(len(support_images)))
            file_obj.flush()

        out_file_path = "{}/{}.npy".format(query_dir, "query")
        count_file_path = "{}/count.txt".format(query_dir)
        print("write to {}".format(out_file_path))
        fp = np.memmap(out_file_path, dtype='float32', mode='w+', shape=query_images.shape)
        fp[:, :, :, :] = query_images[:, :, :, :]
        del fp
        with open(count_file_path, "w") as file_obj:
            file_obj.write(str(len(query_images)))
            file_obj.flush()


def split_test_PART_attack_type(npz_folder, detector):
    extract_pattern = re.compile("(.*)_(.*?)@det_(.*?)@protocol_(.*?)@shot_(\d+)@white_box.npz")
    collect_data = defaultdict(dict)
    for npz_path in os.listdir(npz_folder):
        if npz_path.endswith(".npz"):
            ma = extract_pattern.match(npz_path)
            attack_name = ma.group(1)
            if attack_name == "PGD":
                attack_name = "PGD_L_infinity"
            dataset = ma.group(2)
            detector_name = ma.group(3)
            shot = ma.group(5)
            data = np.load(npz_folder + "/" + npz_path)
            adv_images = data["adv_images"]
            adv_label = data["adv_label"]  # 1 real  0 adv
            gt_label = data["gt_label"]  # 0~ 9 image class label

            collect_data[attack_name+"#"+shot]["adv_images"] = adv_images
            collect_data[attack_name+"#"+shot]["adv_label"] = adv_label
            collect_data[attack_name+"#"+shot]["gt_label"] = gt_label

    for attack_name, data_json in collect_data.items():
        attack_name, shot = attack_name.split("#")
        if attack_name!="clean":
            if detector.startswith("MetaAdvDet"):
                output_root = npz_folder + "/{}@shot_{}/II".format(attack_name, shot)
            else:
                output_root = npz_folder + "/{}/II".format(attack_name)
            os.makedirs(output_root, exist_ok=True)  # /home1/machen/dataset/CIFAR-10/adversarial_images/white_box@data_conv3@det_DNN/FGSM/II
            process_data(attack_name, data_json, output_root)
            process_data("clean", collect_data["clean#1"], output_root)

def split_all_test():
    data_root_dir = {"FashionMNIST":FMNIST_OUTPUT_DATA_DIR,
                     "MNIST":MNIST_OUTPUT_DATA_DIR, "CIFAR-10":CIFAR10_OUTPUT_DATA_DIR}
    detectors = ["DNN", "MetaAdvDet", "NeuralFP", "RotateDet"]
    detectors = ["MetaAdvDetAdvTrain"]
    for dataset, data_root in data_root_dir.items():
        for detector in detectors:
            out_root_dir = data_root + "/white_box@data_conv3@det_{}".format(detector)
            split_test_PART_attack_type(out_root_dir, detector)
            print("dataset:{} and detector:{} is done".format(dataset, detector))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch white box partitioned data generator')

    parser.add_argument("--dataset", type=str, default="CIFAR-10",
                        help="the dataset to train")
    parser.add_argument("--detector", default="DNN",type=str, choices=["DNN", "MetaAdvDet", "MetaAdvDetAdvTrain", "NeuralFP","RotateDet"])
    args = parser.parse_args()

    dataset = args.dataset
    if dataset == "CIFAR-10":
        out_root_dir = CIFAR10_OUTPUT_DATA_DIR
    elif dataset == "CIFAR-100":
        out_root_dir = CIFAR100_OUTPUT_DATA_DIR
    elif dataset == "MNIST":
        out_root_dir = MNIST_OUTPUT_DATA_DIR
    elif dataset == "FashionMNIST":
        out_root_dir = FMNIST_OUTPUT_DATA_DIR
    elif dataset == "SVHN":
        out_root_dir = SVHN_OUTPUT_DATA_DIR
    out_root_dir = out_root_dir + "/white_box@data_conv3@det_{}".format(args.detector)
    assert os.path.exists(out_root_dir), "{} not exists!".format(out_root_dir)
    split_test_PART_attack_type(out_root_dir, args.detector)

    # split_all_test()

