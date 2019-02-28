import sys
sys.path.append("D:/work/adversarial_example")
from cleverhans.generate_adv_script.config import CIFAR10_OUTPUT_DATA_DIR,CIFAR100_OUTPUT_DATA_DIR,\
                        MNIST_OUTPUT_DATA_DIR, FMNIST_OUTPUT_DATA_DIR, \
                        META_ATTACKER_INDEX,META_ATTACKER_PART_I,META_ATTACKER_PART_II, ROOT_DATA_DIR, SVHN_OUTPUT_DATA_DIR
import os
import numpy as np
from collections import defaultdict
import random
import argparse

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
            split_type = 1
            if attack_name in META_ATTACKER_PART_I and attack_name not in META_ATTACKER_PART_II:
                split_type = 1
            elif attack_name not in META_ATTACKER_PART_I and attack_name in META_ATTACKER_PART_II:
                split_type = 2
            elif attack_name in META_ATTACKER_PART_I and attack_name in META_ATTACKER_PART_II:
                split_type = 3

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
            split_type = 1
            if attack_name in META_ATTACKER_PART_I and attack_name not in META_ATTACKER_PART_II:
                split_type = 1
            elif attack_name not in META_ATTACKER_PART_I and attack_name in META_ATTACKER_PART_II:
                split_type = 2
            elif attack_name in META_ATTACKER_PART_I and attack_name in META_ATTACKER_PART_II:
                split_type = 3


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Meta_SGD Training')

    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100", "MNIST", "FMNIST"],
                        help="the dataset to train")
    parser.add_argument("--adv_arch", default="shallow_4_convs",type=str, choices=["shallow_10_convs", "shallow_4_convs", "vgg16small"])
    args = parser.parse_args()
    dataset = args.dataset
    if dataset == "CIFAR10":
        out_root_dir = CIFAR10_OUTPUT_DATA_DIR
    elif dataset == "CIFAR100":
        out_root_dir = CIFAR100_OUTPUT_DATA_DIR
    elif dataset == "MNIST":
        out_root_dir = MNIST_OUTPUT_DATA_DIR
    elif dataset == "FMNIST":
        out_root_dir = FMNIST_OUTPUT_DATA_DIR
    elif dataset == "SVHN":
        out_root_dir = SVHN_OUTPUT_DATA_DIR
    out_root_dir = out_root_dir + "/{}".format(args.adv_arch)
    split_train_PART_attack_type(out_root_dir, "{}/train".format(out_root_dir))
    split_test_PART_attack_type(out_root_dir,  "{}/test".format(out_root_dir))


