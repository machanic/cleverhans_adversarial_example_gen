import os
import shutil
from cleverhans.generate_adv_script.config import META_ATTACKER_PART_I, META_ATTACKER_INDEX


def move_folder(src_folder, dest_folder):
    for folder in os.listdir(src_folder):
        class_id, attack_id = folder.split("_")
        attack_id =  int(attack_id)
        attack_id = attack_id - 1
        attack_name = META_ATTACKER_INDEX[attack_id]
        if attack_name not in META_ATTACKER_PART_I:
            print(src_folder + "/" + folder, dest_folder + "/" + folder)
            shutil.move(src_folder + "/" + folder, dest_folder + "/" + folder)

if __name__ == "__main__":
    move_folder("/home1/machen/dataset/CIFAR-10/adversarial_images/conv4/train/I",
                "/home1/machen/dataset/CIFAR-10/adversarial_images/conv4/train/II")