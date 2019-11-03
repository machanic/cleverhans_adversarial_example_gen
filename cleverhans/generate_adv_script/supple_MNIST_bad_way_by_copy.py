import random

import numpy as np
import os
import copy

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


def get_bad_file_list(root_dir, image_size=224, input_channels=3):
    for root, dirs, files in os.walk(root_dir,True):
        for filename in files:
            file_path = os.path.join(root, filename)
            if file_path.endswith(".txt"):
                file_obj =  open(file_path,"r")
                count = int(file_obj.read())
                file_obj.close()
                if "train" in file_path:
                    total = 40
                elif "support" in file_path:
                    total = 10
                elif "query" in file_path:
                    total = 15
                if count < total:
                    rest = total - count
                    npy_path = list(filter(lambda e:e.endswith(".npy"), os.listdir(os.path.dirname(file_path))))[0]
                    npy_path = os.path.dirname(file_path)+ "/"+npy_path
                    fobj = open(npy_path, "rb")
                    im_list = []
                    all_list = []
                    for idx in range(rest):
                        image_idx = random.randint(0,count-1)
                        im = np.memmap(fobj, dtype='float32', mode='r', shape=(
                            1, image_size, image_size, input_channels),
                                       offset=image_idx * image_size * image_size * input_channels * 32 // 8).copy()
                        im_list.append(im)
                    for idx in range(count):

                        im = np.memmap(fobj, dtype='float32', mode='r', shape=(
                            1, image_size, image_size, input_channels),
                                       offset=idx * image_size * image_size * input_channels * 32 // 8).copy()
                        all_list.append(im)
                    fobj.close()
                    im1 = np.concatenate(all_list,axis=0)
                    im2 = np.concatenate(im_list,axis=0)
                    all_im = np.concatenate([im1,im2],axis=0)
                    fp = np.memmap(npy_path, dtype='float32', mode='write', shape=all_im.shape)
                    fp[:, :, :, :] = all_im[:, :, :, :]
                    del fp
                    with open(file_path, "w") as file_obj:
                        file_obj.write(str(len(all_im)))
                        file_obj.flush()
                    print("write to {} len={}".format(npy_path, len(all_im)))


get_bad_file_list("/home1/machen/dataset/miniimagenet/adversarial_images/resnet10/TRAIN_I_TEST_II/test/",image_size=224,input_channels=3)
# get_bad_file_list("/home1/machen/dataset/MNIST/adversarial_images/resnet18/TRAIN_I_TEST_II")
# get_bad_file_list("/home1/machen/dataset/F-MNIST/adversarial_images/resnet10/TRAIN_I_TEST_II")
# get_bad_file_list("/home1/machen/dataset/F-MNIST/adversarial_images/resnet18/TRAIN_I_TEST_II")