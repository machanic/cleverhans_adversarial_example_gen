"""
This tutorial shows how to generate adversarial examples using FGSM
and train a model using adversarial training with TensorFlow.
The original paper can be found at:
https://arxiv.org/abs/1412.6572
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
sys.path.append("/home1/machen/adversarial_example")

import logging
import numpy as np
import tensorflow as tf

from cleverhans.augmentation import random_horizontal_flip, random_shift
from cleverhans.compat import flags
# from tensorflow.python.platform import flags
from cleverhans.generate_adv_script.config import DATASET_INCHANNELS, DATASET_ADV_OUTPUT,DATASET_SOURCE_PATH,TF_CLEAN_IMAGE_MODEL_PATH
from cleverhans.dataset import CIFAR10, MNIST, CIFAR100, ImageNet, SVHN, AWA2, CUB200_2011
from cleverhans.loss import CrossEntropy
from cleverhans.model_zoo.shallow_CNN import Shallow10ConvLayersConv, Shallow4ConvLayersConv
from cleverhans.model_zoo.vgg import VGG16, VGG16Small
from cleverhans.model_zoo.resnet import ResNet10, ResNet18
from cleverhans.train import train
from cleverhans.utils import AccuracyReport, set_log_level, other_classes
from cleverhans.utils_tf import model_eval, untargeted_advx_image_eval
from cleverhans.generate_adv_script.config import *
import random
from cleverhans.utils_tf import look_for_target_otherthan_gt
import multiprocessing as mp
import os
from cleverhans.generate_adv_script.vgg_preprocessing import preprocess_image


FLAGS = flags.FLAGS


def generate_adv_images(gpu, attack_algo, dataset, source_data_dir, train_start=0, train_end=1000000, test_start=0,
                        test_end=100000, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                        learning_rate=0.001,
                        testing=False,
                         num_threads=None,
                        label_smoothing=0.1, args=FLAGS):
    """
    CIFAR10 cleverhans tutorial
    :param source_data_dir: the CIFAR-10 source data directory
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param clean_train: perform normal training on clean examples only
                        before performing adversarial training.
    :param testing: if true, complete an AccuracyReport for unit tests
                    to verify that performance is adequate
    :param label_smoothing: float, amount of label smoothing for cross entropy
    :return: an AccuracyReport object
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    if "batch_size" in UNTARGETED_ATTACKER_PARAM[attack_algo]:
        global BATCH_SIZE
        batch_size = UNTARGETED_ATTACKER_PARAM[attack_algo]["batch_size"]
        BATCH_SIZE = batch_size
    output_dir = DATASET_ADV_OUTPUT[args.dataset] + "/" + args.arch
    os.makedirs(output_dir, exist_ok=True)
    report = AccuracyReport()
    # if (os.path.exists(output_dir + "/{0}_untargeted_train.npz".format(attack_algo)) and
    #     os.path.exists(output_dir + "/{0}_untargeted_test.npz".format(attack_algo))):
    #     return report


    # Object used to keep track of (and return) key accuracies


    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Create TF session
    config_args = {}
    if num_threads:
        config_args = dict(intra_op_parallelism_threads=1)
    config_args["gpu_options"] = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(**config_args))
    # Get CIFAR10 data
    if dataset == "CIFAR10":
        data = CIFAR10(data_dir=source_data_dir, train_start=train_start, train_end=train_end,
                     test_start=test_start, test_end=test_end)
    elif dataset == "CIFAR100":
        data = CIFAR100(data_dir=source_data_dir, train_start=train_start, train_end=train_end,
                       test_start=test_start, test_end=test_end)
    elif dataset == "MNIST" or dataset == "F-MNIST":
        data = MNIST(data_dir=source_data_dir, train_start=train_start, train_end=train_end, test_start=test_start,
                     test_end=test_end)
    elif dataset == "ImageNet":
        data = ImageNet(data_dir=source_data_dir, train_start=train_start, train_end=train_end, test_start=test_start)
    elif dataset == "SVHN":
        data = SVHN(data_dir=source_data_dir)
    elif dataset == "AWA2":
        data = AWA2(data_dir=source_data_dir)
    elif dataset == "CUB":
        data = CUB200_2011(data_dir=source_data_dir)

    dataset_size = data.x_train.shape[0]
    dataset_train = data.to_tensorflow()[0]
    dataset_train = dataset_train.map(
        lambda x, y: (random_shift(random_horizontal_flip(x)), y), 4)
    dataset_train = dataset_train.batch(batch_size)
    dataset_train = dataset_train.prefetch(16)
    x_train, y_train = data.get_set('train')
    x_test, y_test = data.get_set('test')

    # Use Image Parameters
    img_rows, img_cols, nchannels = x_test.shape[1:4]
    nb_classes = y_test.shape[1]

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, img_rows, img_cols,
                                          nchannels))

    y = tf.placeholder(tf.float32, shape=(BATCH_SIZE, nb_classes))

    if dataset == "ImageNet":

        x = preprocess_image(x, 224, 224, is_training=False)

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    eval_params = {'batch_size': batch_size}

    rng = np.random.RandomState([2017, 8, 30])

    def do_generate_eval(adv_x, pred_adv_x, x_set, y_set, report_key, is_adv=None):
        adv_images_total, adv_pred_total, gt_label_total, success_rate = untargeted_advx_image_eval(sess, x, y, adv_x,
                                                                                                    pred_adv_x, x_set,
                                                                                                    y_set,
                                                                                                    args=eval_params)

        setattr(report, report_key, success_rate)
        if is_adv is None:
            report_text = None
        elif is_adv:
            report_text = 'adversarial'
        else:
            report_text = 'legitimate'
        if report_text:
            print('adversarial attack successful rate on %s: %0.4f' % (report_text, success_rate))
        return adv_images_total, adv_pred_total, gt_label_total, success_rate  # shape = (total, H,W,C)

    def do_eval(preds, x_set, y_set, report_key, is_adv=None):
        acc = model_eval(sess, x, y, preds, x_set, y_set, args=eval_params)
        setattr(report, report_key, acc)
        if is_adv is None:
            report_text = None
        elif is_adv:
            report_text = 'adversarial'
        else:
            report_text = 'legitimate'
        if report_text:
            print('Test accuracy on %s examples: %0.4f' % (report_text, acc))
    if args.arch == "conv4":
        model = Shallow4ConvLayersConv(args.arch, IMG_SIZE[dataset], CLASS_NUM[dataset],
                                       in_channels=DATASET_INCHANNELS[args.dataset], dim_hidden=64)
        model.is_training = False
    elif args.arch == "conv10":
        model = Shallow10ConvLayersConv(args.arch, CLASS_NUM[dataset], nb_filters=64,
                                        input_shape=[IMG_SIZE[dataset], IMG_SIZE[dataset], DATASET_INCHANNELS[args.dataset]])
    elif args.arch == "vgg16":
        model = VGG16("vgg_16", CLASS_NUM[dataset], [IMG_SIZE[dataset], IMG_SIZE[dataset], DATASET_INCHANNELS[args.dataset]])
        model.is_training = False
    elif args.arch == "vgg16small":
        model = VGG16Small(args.arch, CLASS_NUM[dataset], [IMG_SIZE[dataset], IMG_SIZE[dataset], DATASET_INCHANNELS[args.dataset]])
    elif args.arch == "resnet10":
        model = ResNet10(args.arch, CLASS_NUM[dataset], [IMG_SIZE[dataset], IMG_SIZE[dataset], DATASET_INCHANNELS[args.dataset]])
    elif args.arch == "resnet18":
        model = ResNet18(args.arch, CLASS_NUM[dataset],
                         [IMG_SIZE[dataset], IMG_SIZE[dataset], DATASET_INCHANNELS[args.dataset]])



    def evaluate():
        if hasattr(model, "is_training"):
            model.is_training = False
        preds = model.get_logits(x)  # tf.tensor
        do_eval(preds, x_test, y_test, 'clean_train_clean_eval', False)
        if hasattr(model, "is_training"):
            model.is_training = True

    resume = TF_CLEAN_IMAGE_MODEL_PATH[args.dataset] + "/{0}".format(args.arch)
    os.makedirs(resume, exist_ok=True)


    print("using folder {} to store model".format(resume))
    resume_files = os.listdir(resume)
    loss = CrossEntropy(model, smoothing=label_smoothing)
    if len(resume_files) == 0:  # clean train must be done!

        if hasattr(model, "is_training"):
            model.is_training = True

        assert dataset != "ImageNet"  # ImageNet is so big, so we just load pretrained model
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars

        saver = tf.train.Saver(var_list=var_list)
        train(sess, loss, None, None,model,
              dataset_train=dataset_train, dataset_size=dataset_size,
              evaluate=evaluate, args=train_params, rng=rng,
              var_list=model.get_params())  # 训练nb_epochs个epochs
        save_path = saver.save(sess, "{}/model".format(resume), global_step=nb_epochs)
        print("Model saved in path: %s" % save_path)
    else:
        if len(os.listdir(resume)) == 1 and os.listdir(resume)[0].endswith("ckpt"):
            path = resume + "/" + os.listdir(resume)[0]
            var_list = tf.trainable_variables()
            g_list = tf.global_variables()
            bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
            var_list += bn_moving_vars
            saver = tf.train.Saver(var_list=var_list)
            saver.restore(sess, path)

        else:
            # resume from old
            latest_checkpoint = tf.train.latest_checkpoint(resume)
            var_list = tf.trainable_variables()
            g_list = tf.global_variables()
            bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
            var_list += bn_moving_vars
            saver = tf.train.Saver(var_list=var_list)
            saver.restore(sess, latest_checkpoint)



        # Calculate training error
        if testing:
            evaluate()
    if hasattr(model, "is_training"):
        model.is_training = False
    # Initialize the Fast Gradient Sign Method (FGSM) attack object and
    # graph
    attacker = ATTACKERS[attack_algo](model, sess=sess)
    param_dict = UNTARGETED_ATTACKER_PARAM[attack_algo]

    if attack_algo in NEED_TARGETED_Y:
        y_target = look_for_target_otherthan_gt(y, CLASS_NUM[args.dataset])
        y_target = tf.reshape(y_target, (BATCH_SIZE, -1))
        param_dict["y_target"] = y_target

    adv_x = attacker.generate(x, **param_dict)  # tensor
    preds_adv = model.get_logits(adv_x)
    # generate adversarial examples
    adv_images_total, adv_pred_total, gt_label_total, success_rate = do_generate_eval(adv_x, preds_adv, x_train,
                                                                                      y_train,
                                                                                      "clean_train_adv_eval", True)
    np.savez(output_dir + "/{0}_untargeted_train.npz".format(attack_algo), adv_images=adv_images_total,
             adv_pred=adv_pred_total, gt_label=gt_label_total, attack_success_rate=success_rate)

    adv_images_total, adv_pred_total, gt_label_total, success_rate = do_generate_eval(adv_x, preds_adv, x_test,
                                                                                      y_test, "clean_test_adv_eval",
                                                                                      True)
    np.savez(output_dir + "/{0}_untargeted_test.npz".format(attack_algo), adv_images=adv_images_total,
             adv_pred=adv_pred_total,
             gt_label=gt_label_total, attack_success_rate=success_rate)
    print('generate {} adversarial image done'.format(attack_algo))

    return report


def main(argv=None):

    global BATCH_SIZE
    if FLAGS.batch_size != BATCH_SIZE:
        BATCH_SIZE = FLAGS.batch_size

    # if FLAGS.attack == "all":
    #     pool = mp.Pool(processes=len(META_ATTACKER_INDEX))
    #     for index, attack_type in enumerate(META_ATTACKER_INDEX):
    #         gpu = str(FLAGS.gpus).split(",")[index % len(str(FLAGS.gpus).split(","))]
    #         pool.apply_async(generate_adv_images, args=(gpu, attack_type,
    #                             FLAGS.dataset,FLAGS.source_data_dir),kwds=dict(nb_epochs=FLAGS.nb_epochs,
    #                             batch_size=FLAGS.batch_size,
    #                             learning_rate=FLAGS.learning_rate,
    #                             testing=False, args=FLAGS))
    #
    #     pool.close()
    #     pool.join()
    # else:
    generate_adv_images(gpu=FLAGS.gpus, attack_algo=FLAGS.attack, dataset=FLAGS.dataset,
                        source_data_dir=DATASET_SOURCE_PATH[FLAGS.dataset], nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                            learning_rate=FLAGS.learning_rate,
                         testing=True, args=FLAGS)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_filters', NB_FILTERS,
                         'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                         'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', BATCH_SIZE,
                         'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001,
                       'Learning rate for training')
    flags.DEFINE_string('gpus', "0",
                         'GPU for training')
    flags.DEFINE_enum("attack", "FGSM",
                      META_ATTACKER_INDEX, "the attack method")
    flags.DEFINE_enum("dataset", "CIFAR10", ["CIFAR10", "CIFAR100", "MNIST", "F-MNIST", "ImageNet","SVHN", "AWA2","CUB"], "the dataset we want to generate")
    flags.DEFINE_enum("arch", "conv4", ["conv10","conv4", "vgg16","vgg16small", "resnet10", "resnet18"], "the network be used to generate adversarial examples")
    tf.app.run()
