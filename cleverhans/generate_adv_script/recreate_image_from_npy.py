import numpy as np
import os
import cv2
import copy
import random
import re
def recreate_image(x):
    """
        Recreates images from a torch Tensor, sort of reverse preprocessing

    Args:
        x (np.array): C,H,W format Image to recreate

    returns:
        recreated_im (numpy arr): H,W,C format Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    in_channel = x.shape[-1]
    recreated_im = copy.copy(x)  # C, H, W
    if in_channel == 3:
        for c in range(in_channel):
            recreated_im[:, :, c] /= reverse_std[c]
            recreated_im[:, :, c] -= reverse_mean[c]
    elif in_channel == 1:
        recreated_im[:, :, 0] /= reverse_std[1]
        recreated_im[:, :, 0] -= reverse_mean[1]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im)  # H, W, C
    return recreated_im



def recreate_npy(npy_path, output_dir):
    adv_images = np.load(npy_path)["adv_images"]
    select_idx = random.sample(list(np.arange(adv_images.shape[0])), 10)
    adv_images = adv_images[select_idx]
    for idx, adv_image in enumerate(adv_images):
        adv_image = recreate_image(adv_image)
        target_path = output_dir + "/{}.png".format(idx)
        cv2.imwrite(target_path, adv_image)
        print("write {} done".format(target_path))

if __name__ == "__main__":
    extract_noise_type_pattern = re.compile("(.*?)_untargeted_train.npz")
    for dataset in ["MNIST", "F-MNIST",]:
        ROOT_DIR_PATH = "/home1/machen/dataset/{}/adversarial_images/".format(dataset)

        for arch in os.listdir(ROOT_DIR_PATH):
            for npy_path in os.listdir(ROOT_DIR_PATH + "/" +arch):
                if npy_path.endswith("npz") and npy_path.endswith("train.npz"):
                    ma = extract_noise_type_pattern.match(npy_path)
                    noise_type = ma.group(1)
                    npy_path = ROOT_DIR_PATH + "/" + arch + "/" + npy_path
                    target_folder = npy_path.replace("adversarial_images", "adv_images_png")
                    target_folder = os.path.dirname(target_folder) + "/" + noise_type
                    os.makedirs(target_folder, exist_ok=True)
                    if dataset in ["MNIST", "F-MNIST"]:
                        npy_path = npy_path.replace("adversarial_images","processed_adv_img")
                    recreate_npy(npy_path, target_folder)