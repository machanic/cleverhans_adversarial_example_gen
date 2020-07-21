import sys
sys.path.append("/home1/machen/adversarial_example")
from cleverhans.generate_adv_script.config import CIFAR10_OUTPUT_DATA_DIR, MNIST_OUTPUT_DATA_DIR, FMNIST_OUTPUT_DATA_DIR, \
    META_ATTACKER_INDEX, META_ATTACKER_PART_I, META_ATTACKER_PART_II, SVHN_OUTPUT_DATA_DIR, \
    ILSVRC12_OUTPUT_DATA_DIR, CIFAR100_COARSE_LABEL_OUTPUT_DATA_DIR
import os
import numpy as np
from collections import defaultdict
import random
import argparse
import glob
import re

def split_train(npz_folder, output_root_dir):
    for npz_path in os.listdir(npz_folder):
        if npz_path.endswith(".npz") and "train" in npz_path:
            attack_name = npz_path[:npz_path.index("_untargeted")]
            if attack_name in META_ATTACKER_INDEX:
                attack_index = META_ATTACKER_INDEX.index(attack_name) + 1   # clean: 1
                npz_path = npz_folder + "/" + npz_path
                data = np.load(npz_path)
                adv_images = data["adv_images"]
                adv_pred= data["adv_pred"]
                gt_label = data["gt_label"]
                sucess_rate = data["attack_success_rate"]
                print(sucess_rate)
                if attack_index == 1:
                    indexes = np.arange(adv_images.shape[0])
                else:
                    indexes = np.where(adv_pred != gt_label)[0]
                adv_images = adv_images[indexes]
                gt_label = gt_label[indexes]
                adv_image_dict = defaultdict(list)
                for idx, label in enumerate(gt_label):
                    adv_image_dict[label].append(adv_images[idx])    # key = gt_label, value = image list
                for label, adv_images_list in adv_image_dict.items():
                    out_dir_name = "{}/{}_{}".format(output_root_dir, label, attack_index)
                    os.makedirs(out_dir_name, exist_ok=True)
                    out_file_path = "{}/{}.npy".format(out_dir_name, "train")
                    count_file_path = "{}/count.txt".format(out_dir_name)
                    adv_images = np.stack(adv_images_list)
                    fp = np.memmap(out_file_path, dtype='float32', mode='w+', shape=adv_images.shape)
                    fp[:,:,:,:] = adv_images[:,:,:,:]
                    del fp
                    with open(count_file_path, "w") as file_obj:
                        file_obj.write(str(len(adv_images)))
                        file_obj.flush()
                    print("save {} image files into {}".format(len(adv_images),out_file_path))

def split_train_PART_attack_type(npz_folder, output_root_dir):
    for npz_path in os.listdir(npz_folder):
        if npz_path.endswith(".npz") and "train" in npz_path:
            attack_name = npz_path[:npz_path.index("_untargeted")]
            split_type = 0
            if attack_name in META_ATTACKER_PART_I and attack_name not in META_ATTACKER_PART_II:
                split_type = 1
            elif attack_name not in META_ATTACKER_PART_I and attack_name in META_ATTACKER_PART_II:
                split_type = 2
            elif attack_name in META_ATTACKER_PART_I and attack_name in META_ATTACKER_PART_II:
                split_type = 3
            if split_type == 0:
                continue
            attack_index = META_ATTACKER_INDEX.index(attack_name) + 1   # clean: 1
            npz_path = npz_folder + "/" + npz_path
            data = np.load(npz_path)
            adv_images = data["adv_images"]
            adv_pred= data["adv_pred"]
            gt_label = data["gt_label"]
            sucess_rate = data["attack_success_rate"]
            print(sucess_rate)
            if attack_index == 1: # clean data
                indexes = np.arange(adv_images.shape[0])
            else:
                indexes = np.where(adv_pred != gt_label)[0]
            adv_images = adv_images[indexes]
            gt_label = gt_label[indexes]
            adv_image_dict = defaultdict(list)
            for idx, label in enumerate(gt_label):  # FIXME 修改为adv_pred
                adv_image_dict[label].append(adv_images[idx])    # key = gt_label, value = image list
            for label, adv_images_list in adv_image_dict.items():
                out_dir_names = []
                if split_type == 1:
                    out_dir_names = ["{}/I/{}_{}".format(output_root_dir, label, attack_index)]
                elif split_type == 2:
                    out_dir_names = ["{}/II/{}_{}".format(output_root_dir, label, attack_index)]
                elif split_type == 3:
                    out_dir_names = ["{}/I/{}_{}".format(output_root_dir, label, attack_index),
                                     "{}/II/{}_{}".format(output_root_dir, label, attack_index)]
                for out_dir_name in out_dir_names:
                    os.makedirs(out_dir_name, exist_ok=True)
                    out_file_path = "{}/{}.npy".format(out_dir_name, "train")
                    count_file_path = "{}/count.txt".format(out_dir_name)
                    adv_images = np.stack(adv_images_list)
                    fp = np.memmap(out_file_path, dtype='float32', mode='w+', shape=adv_images.shape)
                    fp[:,:,:,:] = adv_images[:,:,:,:]
                    del fp
                    with open(count_file_path, "w") as file_obj:
                        file_obj.write(str(len(adv_images)))
                        file_obj.flush()
                    print("save {} image files into {}".format(len(adv_images),out_file_path))


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


def split_test_PART_attack_type(npz_folder, output_root_dir):
    for npz_path in os.listdir(npz_folder):
        if npz_path.endswith(".npz") and "test" in npz_path:

            attack_name = npz_path[:npz_path.index("_untargeted")]
            split_type = 0
            if attack_name in META_ATTACKER_PART_I and attack_name not in META_ATTACKER_PART_II:
                split_type = 1
            elif attack_name not in META_ATTACKER_PART_I and attack_name in META_ATTACKER_PART_II:
                split_type = 2
            elif attack_name in META_ATTACKER_PART_I and attack_name in META_ATTACKER_PART_II:
                split_type = 3
            if split_type == 0:
                continue

            if attack_name in META_ATTACKER_INDEX:
                attack_index = META_ATTACKER_INDEX.index(attack_name) + 1   # clean: 1
                npz_path = npz_folder + "/" + npz_path
                data = np.load(npz_path)
                adv_images = data["adv_images"]
                adv_pred= data["adv_pred"]
                gt_label = data["gt_label"]
                sucess_rate = data["attack_success_rate"]
                print(sucess_rate)
                if attack_index == 1:
                    indexes = np.arange(adv_images.shape[0])
                else:
                    indexes = np.where(adv_pred != gt_label)[0]
                adv_images = adv_images[indexes]
                gt_label = gt_label[indexes]
                adv_image_dict = defaultdict(list)
                for idx, label in enumerate(gt_label):
                    adv_image_dict[label].append(adv_images[idx])    # key = gt_label, value = image list
                for label, adv_images_list in adv_image_dict.items():
                    out_dir_names = []
                    if split_type == 1:
                        out_dir_names = ["{}/I/{}_{}".format(output_root_dir, label, attack_index)]
                    elif split_type == 2:
                        out_dir_names = ["{}/II/{}_{}".format(output_root_dir, label, attack_index)]
                    elif split_type == 3:
                        out_dir_names = ["{}/I/{}_{}".format(output_root_dir, label, attack_index),
                                         "{}/II/{}_{}".format(output_root_dir, label, attack_index)]
                    for out_dir_name in out_dir_names:
                        support_dir = "{}/support".format(out_dir_name)
                        query_dir = "{}/query".format(out_dir_name)
                        adv_images = np.stack(adv_images_list)
                        os.makedirs(support_dir, exist_ok=True)
                        os.makedirs(query_dir, exist_ok=True)
                        all_index = np.arange(len(adv_images_list))
                        sub_lists = list(chunk(all_index, 2))
                        support_indexes = sub_lists[0]
                        query_indexes = sub_lists[1]
                        if len(support_indexes) >= 20:
                            support_indexes = random.sample(support_indexes, 20)
                        if len(query_indexes) >= 60:
                            query_indexes = random.sample(query_indexes, 60)
                        support_images = adv_images[support_indexes]
                        query_images = adv_images[query_indexes]
                        out_file_path = "{}/{}.npy".format(support_dir, "support")
                        count_file_path = "{}/count.txt".format(support_dir)
                        fp = np.memmap(out_file_path, dtype='float32', mode='w+', shape=support_images.shape)
                        fp[:,:,:,:] = support_images[:,:,:,:]
                        del fp
                        with open(count_file_path, "w") as file_obj:
                            file_obj.write(str(len(support_images)))
                            file_obj.flush()

                        out_file_path = "{}/{}.npy".format(query_dir, "query")
                        count_file_path = "{}/count.txt".format(query_dir)
                        fp = np.memmap(out_file_path, dtype='float32', mode='w+', shape=query_images.shape)
                        fp[:, :, :, :] = query_images[:, :, :, :]
                        del fp
                        with open(count_file_path, "w") as file_obj:
                            file_obj.write(str(len(query_images)))
                            file_obj.flush()

                        print("save {} image files into {}".format(len(query_images),out_file_path))
                adv_image_dict.clear()
                adv_images = None
                del data


def save_npz_file(npz_path, leave_one_attack_name, attack_index, output_root_dir, train_test_str):
    data = np.load(npz_path)
    adv_images = data["adv_images"]
    adv_pred = data["adv_pred"]
    gt_label = data["gt_label"]
    sucess_rate = data["attack_success_rate"]
    print(sucess_rate)
    if attack_index == 1:  # clean data
        indexes = np.arange(adv_images.shape[0])
    else:
        indexes = np.where(adv_pred != gt_label)[0]
    adv_images = adv_images[indexes]
    gt_label = gt_label[indexes]
    adv_image_dict = defaultdict(list)
    for idx, label in enumerate(gt_label):
        adv_image_dict[label].append(adv_images[idx])  # key = gt_label, value = image list

    for label, adv_images_list in adv_image_dict.items():
        if train_test_str == "train":
            out_dir_name = "{}/{}/{}/{}_{}".format(output_root_dir, leave_one_attack_name, train_test_str,
                                                   label, attack_index)
            os.makedirs(out_dir_name, exist_ok=True)
            out_file_path = "{}/{}.npy".format(out_dir_name, "train")
            count_file_path = "{}/count.txt".format(out_dir_name)
            adv_images = np.stack(adv_images_list)
            fp = np.memmap(out_file_path, dtype='float32', mode='w+', shape=adv_images.shape)
            fp[:, :, :, :] = adv_images[:, :, :, :]
            del fp
            with open(count_file_path, "w") as file_obj:
                file_obj.write(str(len(adv_images)))
                file_obj.flush()
            print("save {} image files into {}".format(len(adv_images), out_file_path))
        else:
            out_dir_name = "{}/{}/{}/{}_{}".format(output_root_dir, leave_one_attack_name, train_test_str,
                                                   label, attack_index)
            all_index = np.arange(len(adv_images_list))
            sub_lists = list(chunk(all_index, 2))
            support_indexes = sub_lists[0]
            query_indexes = sub_lists[1]
            if len(support_indexes) >= 30:
                support_indexes = random.sample(support_indexes, 30)
            if len(query_indexes) >= 60:
                query_indexes = random.sample(query_indexes, 60)
            support_images = adv_images[support_indexes]
            query_images = adv_images[query_indexes]

            for post_str in ["support", "query"]:
                if post_str=='support':
                    current_imgs = support_images
                else:
                    current_imgs = query_images
                dir_name = "{}/{}".format(out_dir_name, post_str)
                os.makedirs(dir_name,exist_ok=True)
                out_file_path = dir_name +'/' + post_str +".npy"
                fp = np.memmap(out_file_path, dtype='float32', mode='w+', shape=current_imgs.shape)
                fp[:, :, :, :] = current_imgs[:, :, :, :]
                del fp
                count_file_path = "{}/count.txt".format(dir_name)
                with open(count_file_path, "w") as file_obj:
                    file_obj.write(str(len(current_imgs)))
                    file_obj.flush()
                print("save {} image files into {}".format(len(current_imgs), out_file_path))

def split_leave_one_out_attack_type(npz_folder, output_root_dir):

    total_train_npz_path_list = glob.glob("{}/*train.npz".format(npz_folder))
    extract_attack_name_pattern = re.compile(".*/(.*?)_untargeted.*")
    for test_npz_path in glob.glob("{}/*test.npz".format(npz_folder)):  # leave one out protocol
        ma = extract_attack_name_pattern.match(test_npz_path)
        leave_out_attack_name = ma.group(1)
        if leave_out_attack_name == "clean":  # clean同时存在于训练集和测试集
            continue
        train_npz_path_list = [path for path in total_train_npz_path_list if leave_out_attack_name not in path]
        test_npz_path_list = [test_npz_path, "{}/clean_untargeted_test.npz".format(npz_folder)]
        for train_npz_path in train_npz_path_list:
            ma = extract_attack_name_pattern.match(train_npz_path)
            current_atk_index = META_ATTACKER_INDEX.index(ma.group(1))+1
            save_npz_file(train_npz_path, leave_out_attack_name, current_atk_index, output_root_dir, "train")
        for test_npz_path in test_npz_path_list:
            ma = extract_attack_name_pattern.match(test_npz_path)
            current_atk_index = META_ATTACKER_INDEX.index(ma.group(1))+1
            save_npz_file(test_npz_path, leave_out_attack_name,current_atk_index, output_root_dir, "test")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Meta_SGD Training')

    parser.add_argument("--dataset", type=str, default="CIFAR-10", choices=["CIFAR-10", "CIFAR-100", "SVHN", "MNIST", "FashionMNIST","ImageNet"],
                        help="the dataset to train")
    parser.add_argument("--adv_arch", default="conv4",type=str, choices=["conv4", "resnet10", "resnet18"])
    args = parser.parse_args()
    dataset = args.dataset
    if dataset == "ImageNet":
        out_root_dir = ILSVRC12_OUTPUT_DATA_DIR
    elif dataset == "CIFAR-10":
        out_root_dir = CIFAR10_OUTPUT_DATA_DIR
    elif dataset == "CIFAR-100":
        out_root_dir = CIFAR100_COARSE_LABEL_OUTPUT_DATA_DIR
    elif dataset == "MNIST":
        out_root_dir = MNIST_OUTPUT_DATA_DIR
    elif dataset == "FashionMNIST":
        out_root_dir = FMNIST_OUTPUT_DATA_DIR
    elif dataset == "SVHN":
        out_root_dir = SVHN_OUTPUT_DATA_DIR
    out_root_dir = out_root_dir + "/{}/npz".format(args.adv_arch)
    # split_train_PART_attack_type(out_root_dir, "{}/TRAIN_I_TEST_II/train".format(out_root_dir))
    split_test_PART_attack_type(out_root_dir, "{}/TRAIN_I_TEST_II/test".format(out_root_dir))

    # split_leave_one_out_attack_type(out_root_dir, "{}/leave_one_out".format(out_root_dir))


