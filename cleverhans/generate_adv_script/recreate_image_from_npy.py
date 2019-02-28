import numpy as np
import os
import cv2

def recreate_image(float_image):
    """
        Recreates images from a torch Tensor, sort of reverse preprocessing

    Args:
        float_image (np.ndarray): N,H,W,C  Image to recreate

    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = np.array([[[[-0.485, -0.456, -0.406]]]])  # 1，1，1，3
    reverse_std = np.array([[[[1/0.229, 1/0.224, 1/0.225]]]])
    float_image /= reverse_std
    float_image -= reverse_mean
    float_image[float_image > 1] = 1
    float_image[float_image < 0] = 0
    recreated_im = np.round(float_image * 255)

    recreated_im = np.uint8(recreated_im)  # N, H, W, C
    return recreated_im

def walk_recreate(folder, output_dir):

    for sub_folder in os.listdir(folder):

        npy_file = folder + '/' + sub_folder + '/' + "train.npy"
        array = np.load(npy_file)
        orig_images = recreate_image(array)
        target_folder = output_dir + "/" + sub_folder
        os.makedirs(target_folder, exist_ok=True)
        for idx, orig_image in enumerate(orig_images):
            cv2.imwrite(target_folder + "/{}.png".format(idx), orig_image)
        print("write {} done".format(sub_folder))

if __name__ == "__main__":
    walk_recreate("/home1/machen/dataset/CIFAR-10/split_data/train", "/home1/machen/dataset/CIFAR-10/output/images")