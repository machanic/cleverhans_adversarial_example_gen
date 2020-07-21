import os
import sys
sys.path.append("/home1/machen/adversarial_example")
import numpy as np
orig_npz_dir = "/home1/machen/dataset/miniimagenet/adversarial_images/resnet18/npz"
root_dir = "/home1/machen/dataset/miniimagenet/adversarial_images/resnet18"
for npz_file in os.listdir(orig_npz_dir):

    data = np.load(orig_npz_dir + "/" + npz_file)
    adv_pred = data["adv_pred"]
    gt_label  = data['gt_label']
    attack_success_rate = data["attack_success_rate"]
    new_random_access_npy_path = root_dir + "/" + npz_file.replace(".npz", ".npy")
    new_additional_info_path = root_dir + "/" + npz_file
    np.savez(new_additional_info_path, adv_pred=adv_pred, gt_label=gt_label, attack_success_rate=attack_success_rate)

    adv_images = data["adv_images"]
    fp = np.memmap(new_random_access_npy_path, dtype='float32', mode='w+', shape=adv_images.shape)
    fp[:, :, :, :] = adv_images[:, :, :, :]
    del fp
    print("{} done".format(npz_file))