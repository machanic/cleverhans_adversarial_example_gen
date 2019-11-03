from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
sys.path.append("/home1/machen/adversarial_example")
import logging
import numpy as np
import tensorflow as tf
from cleverhans.attacks import Attack
from cleverhans.model import Model, wrapper_warning_logits, CallableModelWrapper
from cleverhans.augmentation import random_horizontal_flip, random_shift
from tensorflow.python.platform import app, flags
from cleverhans.dataset import CIFAR
from cleverhans.loss import CrossEntropy
from cleverhans.model_zoo.all_convolutional import ModelAllConvolutional
from cleverhans.train import train
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.utils_tf import model_eval
from cleverhans.attacks import VirtualAdversarialMethod, MomentumIterativeMethod, BasicIterativeMethod
import warnings
from cleverhans.utils import _ArgsWrapper
import math
import os
from cleverhans.attacks_tf import jacobian_graph
from cleverhans import utils_tf
from cleverhans import utils
import copy

FLAGS = flags.FLAGS

# NOTE: I found 4 attack method in Cifar-10 untargeted attack has low successful rate, which are
# 1. VirtualAdversarialMethod, 2. L2 norm of MomentumIterativeMethod,
# 3. L2 norm of BasicIterativeMethod, 4. L infinity version of deep_fool(the original cleverhans deepfool is L2 norm, I extend it to L-inf version according to original paper,

# In addition, I also test a L-infinity version of DeepFool, this code is just little modifed of original cleverhans' DeepFool
# to add L-inf functionality, which is described in original paper's page 5.
# paper can be downloaded from https://arxiv.org/abs/1511.04599.  But it also test FAILED! SUCCESSFUL RATE is so LOW!
# implementation of L-infinity version of DeepFool, just modified little code of original L2 version cleverhans deepfool
_logger = utils.create_logger("cleverhans.attacks.tf")
def deepfool_batch(sess,
                   x,
                   pred,
                   logits,
                   grads,
                   X,
                   nb_candidate,
                   overshoot,
                   max_iter,
                   clip_min,
                   clip_max,
                   nb_classes,
                   Lp_norm=2,
                   feed=None,
                   ):
    """
    Applies DeepFool to a batch of inputs
    :param sess: TF session
    :param x: The input placeholder
    :param pred: The model's sorted symbolic output of logits, only the top
                 nb_candidate classes are contained
    :param logits: The model's unnormalized output tensor (the input to
                   the softmax layer)
    :param grads: Symbolic gradients of the top nb_candidate classes, procuded
                  from gradient_graph
    :param X: Numpy array with sample inputs
    :param nb_candidate: The number of classes to test against, i.e.,
                         deepfool only consider nb_candidate classes when
                         attacking(thus accelerate speed). The nb_candidate
                         classes are chosen according to the prediction
                         confidence during implementation.
    :param overshoot: A termination criterion to prevent vanishing updates
    :param max_iter: Maximum number of iteration for DeepFool
    :param clip_min: Minimum value for components of the example returned
    :param clip_max: Maximum value for components of the example returned
    :param nb_classes: Number of model output classes
    :return: Adversarial examples
    """
    if Lp_norm == 2:
        X_adv = deepfool_attack_L2(
            sess,
            x,
            pred,
            logits,
            grads,
            X,
            nb_candidate,
            overshoot,
            max_iter,
            clip_min,
            clip_max,
            feed=feed)
    else:
        X_adv = deepfool_attack_Lp(sess,
            x,
            Lp_norm,
            pred,
            logits,
            grads,
            X,
            nb_candidate,
            overshoot,
            max_iter,
            clip_min,
            clip_max,
            feed=feed)

    return np.asarray(X_adv, dtype=np.dtype('float32'))


def deepfool_attack_Lp(sess,
                    x,
                    Lp_norm,
                    predictions,
                    logits,
                    grads,
                    sample,
                    nb_candidate,
                    overshoot,
                    max_iter,
                    clip_min,
                    clip_max,
                    feed=None):
    """
    TensorFlow implementation of DeepFool for Lp normalization, which is described in original paper's page 5.
    Paper link: see https://arxiv.org/pdf/1511.04599.pdf  #it is said by <On detecting Adversarial Perturbations>, deepfool has L2 and L_infinity versions
    :param sess: TF session
    :param x: The input placeholder
    :param Lp_norm: The L_p norm in original paper's page 5.
    :param predictions: The model's sorted symbolic output of logits, only the
                       top nb_candidate classes are contained
    :param logits: The model's unnormalized output tensor (the input to
                   the softmax layer)
    :param grads: Symbolic gradients of the top nb_candidate classes, procuded
                 from gradient_graph
    :param sample: Numpy array with sample input
    :param nb_candidate: The number of classes to test against, i.e.,
                         deepfool only consider nb_candidate classes when
                         attacking(thus accelerate speed). The nb_candidate
                         classes are chosen according to the prediction
                         confidence during implementation.
    :param overshoot: A termination criterion to prevent vanishing updates
    :param max_iter: Maximum number of iteration for DeepFool
    :param clip_min: Minimum value for components of the example returned
    :param clip_max: Maximum value for components of the example returned
    :return: Adversarial examples
    """
    adv_x = copy.copy(sample)
    # Initialize the loop variables
    iteration = 0  # paper line 4
    current = utils_tf.model_argmax(sess, x, logits, adv_x, feed=feed)  # 当前的预测label输出
    if current.shape == ():
        current = np.array([current])
    w = np.squeeze(np.zeros(sample.shape[1:]))  # same shape as original image
    r_tot = np.zeros(sample.shape)  # perturbation
    original = current  # use original label as the reference

    _logger.debug(
        "Starting DeepFool attack up to %s iterations", max_iter)
    # Repeat this main loop until we have achieved misclassification
    while (np.any(current == original) and iteration < max_iter):

        # if iteration % 5 == 0 and iteration > 0:
        #     _logger.info("Attack result at iteration %s is %s", iteration, current)
        gradients = sess.run(grads, feed_dict={x: adv_x})
        predictions_val = sess.run(predictions, feed_dict={x: adv_x})
        for idx in range(sample.shape[0]):
            pert = np.inf
            if current[idx] != original[idx]:
                continue
            for k in range(1, nb_candidate):  # paper code line number 6
                w_k = gradients[idx, k, ...] - gradients[idx, 0, ...]
                f_k = predictions_val[idx, k] - predictions_val[idx, 0] # paper code line number 8
                # adding value 0.00001 to prevent f_k = 0
                if Lp_norm == np.inf: # 无穷范数的规则特殊 论文第五页特殊提到了
                    pert_k = (abs(f_k) + 0.00001) / np.linalg.norm(w_k.flatten(), ord=1)
                else:
                    pert_k = (abs(f_k) + 0.00001) / np.linalg.norm(w_k.flatten(), ord=Lp_norm)  # paper code line number 10
                if pert_k < pert: # 论文中的argmin
                    pert = pert_k
                    w = w_k
            q = Lp_norm / (Lp_norm - 1)  # one can apply Holder’s inequality to obtain a lower bound on the Lp norm of the perturbation.
            if Lp_norm == np.inf:
                # pert change to Lp norm
                r_i = (pert + 1e-4) * np.sign(w) / np.linalg.norm(w.flatten(), ord=1)  # N,3,H,W   paper code line number 11, pert 就是论文的分子
            else: # infinity norm attack
                r_i = ((pert + 1e-4) * np.power(np.absolute(w), q - 1) / np.linalg.norm(w.flatten(), ord=q)) * np.sign(w)  # N,3,H,W   paper code line number 11, pert 就是论文的分子
            r_tot[idx, ...] = r_tot[idx, ...] + r_i  # paper code line number 15
        adv_x = np.clip(r_tot + sample, clip_min, clip_max)
        current = utils_tf.model_argmax(sess, x, logits, adv_x, feed=feed)
        if current.shape == ():
            current = np.array([current])
        # Update loop variables
        iteration = iteration + 1

    # need more revision, including info like how many succeed
    # _logger.info("Attack result at iteration %s is %s", iteration, current)
    _logger.info("%s out of %s become adversarial examples at iteration %s",
                 sum(current != original),
                 sample.shape[0],
                 iteration)
    # need to clip this image into the given range
    adv_x = np.clip((1 + overshoot) * r_tot + sample, clip_min, clip_max)
    return adv_x


def deepfool_attack_L2(sess,
                       x,
                       predictions,
                       logits,
                       grads,
                       sample,
                       nb_candidate,
                       overshoot,
                       max_iter,
                       clip_min,
                       clip_max,
                       feed=None):
    """
    TensorFlow implementation of DeepFool.
    Paper link: see https://arxiv.org/pdf/1511.04599.pdf  #it is said by <On detecting Adversarial Perturbations>, deepfool has L2 and L_infinity versions
    :param sess: TF session
    :param x: The input placeholder
    :param predictions: The model's sorted symbolic output of logits, only the
                       top nb_candidate classes are contained
    :param logits: The model's unnormalized output tensor (the input to
                   the softmax layer)
    :param grads: Symbolic gradients of the top nb_candidate classes, procuded
                 from gradient_graph
    :param sample: Numpy array with sample input
    :param nb_candidate: The number of classes to test against, i.e.,
                         deepfool only consider nb_candidate classes when
                         attacking(thus accelerate speed). The nb_candidate
                         classes are chosen according to the prediction
                         confidence during implementation.
    :param overshoot: A termination criterion to prevent vanishing updates
    :param max_iter: Maximum number of iteration for DeepFool
    :param clip_min: Minimum value for components of the example returned
    :param clip_max: Maximum value for components of the example returned
    :return: Adversarial examples
    """
    adv_x = copy.copy(sample)
    # Initialize the loop variables
    iteration = 0
    current = utils_tf.model_argmax(sess, x, logits, adv_x, feed=feed)
    if current.shape == ():
        current = np.array([current])
    w = np.squeeze(np.zeros(sample.shape[1:]))  # same shape as original image
    r_tot = np.zeros(sample.shape)
    original = current  # use original label as the reference

    _logger.debug(
        "Starting DeepFool attack up to %s iterations", max_iter)
    # Repeat this main loop until we have achieved misclassification
    while (np.any(current == original) and iteration < max_iter):

        # if iteration % 5 == 0 and iteration > 0:
        #     _logger.info("Attack result at iteration %s is %s", iteration, current)
        gradients = sess.run(grads, feed_dict={x: adv_x})
        predictions_val = sess.run(predictions, feed_dict={x: adv_x})
        for idx in range(sample.shape[0]):
            pert = np.inf
            if current[idx] != original[idx]:
                continue
            for k in range(1, nb_candidate):
                w_k = gradients[idx, k, ...] - gradients[idx, 0, ...]
                f_k = predictions_val[idx, k] - predictions_val[idx, 0]
                # adding value 0.00001 to prevent f_k = 0
                pert_k = (abs(f_k) + 0.00001) / np.linalg.norm(w_k.flatten())
                if pert_k < pert:
                    pert = pert_k
                    w = w_k
            r_i = pert * w / np.linalg.norm(w.flatten())
            r_tot[idx, ...] = r_tot[idx, ...] + r_i

        adv_x = np.clip(r_tot + sample, clip_min, clip_max)
        current = utils_tf.model_argmax(sess, x, logits, adv_x, feed=feed)
        if current.shape == ():
            current = np.array([current])
        # Update loop variables
        iteration = iteration + 1

    # need more revision, including info like how many succeed
    # _logger.info("Attack result at iteration %s is %s", iteration, current)
    _logger.info("%s out of %s become adversarial examples at iteration %s",
                 sum(current != original),
                 sample.shape[0],
                 iteration)
    # need to clip this image into the given range
    adv_x = np.clip((1 + overshoot) * r_tot + sample, clip_min, clip_max)
    return adv_x

class DeepFool(Attack):
    """
    DeepFool is an untargeted & iterative attack which is based on an
    iterative linearization of the classifier. The implementation here
    is w.r.t. the L2 norm.
    Paper link: "https://arxiv.org/pdf/1511.04599.pdf"

    :param model: cleverhans.model.Model
    :param sess: tf.Session
    :param dtypestr: dtype of the data
    :param kwargs: passed through to super constructor
    """

    def __init__(self, model, sess, dtypestr='float32', **kwargs):
        """
        Create a DeepFool instance.
        """
        if not isinstance(model, Model):
            wrapper_warning_logits()
            model = CallableModelWrapper(model, 'logits')

        super(DeepFool, self).__init__(model, sess, dtypestr, **kwargs)

        self.structural_kwargs = [
            'overshoot', 'max_iter', 'clip_max', 'clip_min', 'nb_candidate', "Lp_norm"
        ]

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.

        :param x: The model's symbolic inputs.
        :param kwargs: See `parse_params`
        """
        assert self.sess is not None, \
            'Cannot use `generate` when no `sess` was provided'

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Define graph wrt to this input placeholder
        logits = self.model.get_logits(x)
        self.nb_classes = logits.get_shape().as_list()[-1]
        assert self.nb_candidate <= self.nb_classes, \
            'nb_candidate should not be greater than nb_classes'
        preds = tf.reshape(
            tf.nn.top_k(logits, k=self.nb_candidate)[0],
            [-1, self.nb_candidate])
        # grads will be the shape [batch_size, nb_candidate, image_size]
        grads = tf.stack(jacobian_graph(preds, x, self.nb_candidate), axis=1)

        # Define graph
        def deepfool_wrap(x_val):
            return deepfool_batch(self.sess, x, preds, logits, grads, x_val,
                                  self.nb_candidate, self.overshoot,
                                  self.max_iter, self.clip_min, self.clip_max,
                                  self.nb_classes, self.Lp_norm)

        wrap = tf.py_func(deepfool_wrap, [x], self.tf_dtype)
        wrap.set_shape(x.get_shape())
        return wrap

    def parse_params(self,
                     nb_candidate=10,
                     overshoot=0.02,
                     max_iter=50,
                     clip_min=0.,
                     clip_max=1.,
                     Lp_norm=2,
                     **kwargs):
        """
        :param nb_candidate: The number of classes to test against, i.e.,
                             deepfool only consider nb_candidate classes when
                             attacking(thus accelerate speed). The nb_candidate
                             classes are chosen according to the prediction
                             confidence during implementation.
        :param overshoot: A termination criterion to prevent vanishing updates
        :param max_iter: Maximum number of iteration for deepfool
        :param clip_min: Minimum component value for clipping
        :param clip_max: Maximum component value for clipping
        :param Lp_norm: default is 2, alternative can be set to np.inf for infinity norm attack
        """
        self.nb_candidate = nb_candidate
        self.overshoot = overshoot
        self.max_iter = max_iter
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.Lp_norm = Lp_norm
        if len(kwargs.keys()) > 0:
            warnings.warn("kwargs is unused and will be removed on or after "
                          "2019-04-26.")

        return True


# implementation of L-inf version of DeepFool over




ATTACKERS = {"VAT": VirtualAdversarialMethod, "MI_FGSM_L2": MomentumIterativeMethod,
             "BIM_L2": BasicIterativeMethod, "deep_fool_L_infinity":DeepFool,
             }
ATTACK_PARAM = {"VAT": {"nb_iter":20, "clip_min":-3., "clip_max":3., "eps":6.0, "xi":1e-6},
                "BIM_L2" :{"ord":2, "rand_init":False, "rand_minmax":0, "nb_iter":40,"eps_iter":0.1, "clip_min":-3., "clip_max":3.},
                "MI_FGSM_L2": {"ord": 2, "eps": 0.3, "nb_iter": 40, "decay_factor": 1.0, "eps_iter": 0.1,
                               "clip_min": -3., "clip_max": 3., },
                "deep_fool_L_infinity":{ "overshoot": 0.02, "max_iter": 40, "clip_min": -3., "clip_max": 3., "batch_size": 500, "Lp_norm": np.inf},
                }
NB_EPOCHS  = 50
BATCH_SIZE = 100
NB_FILTERS = 64
CLEAN_TRAIN = True
LEARNING_RATE = 0.001
CIFAR_MODEL_STORE_PATH = "./trained_CIFAR_model/"
os.makedirs(CIFAR_MODEL_STORE_PATH, exist_ok=True)

_model_eval_cache = {}
def untargeted_advx_image_eval(sess, x, y, adversarial_image, logit_adv_x, X_test=None, Y_test=None,
                               feed=None, args=None):
    global _model_eval_cache
    args = _ArgsWrapper(args or {})

    assert args.batch_size, "Batch size was not given in args dict"
    if X_test is None or Y_test is None:
        raise ValueError("X_test argument and Y_test argument "
                         "must be supplied.")

    # Define accuracy symbolically
    key = (y, logit_adv_x)
    pred_adv_x = tf.argmax(logit_adv_x, axis=-1)
    if key in _model_eval_cache:
        pred_not_equal_orig = _model_eval_cache[key]
    else:

        pred_not_equal_orig = tf.math.logical_not(tf.equal(tf.argmax(y, axis=-1),
                                                           pred_adv_x))
        _model_eval_cache[key] = pred_not_equal_orig

    # Init result var
    success_rate = 0.0

    adv_images_total = []
    adv_pred_total = []
    gt_label_total = []

    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_test)) / args.batch_size))
        assert nb_batches * args.batch_size >= len(X_test)

        X_cur = np.zeros((args.batch_size,) + X_test.shape[1:],
                         dtype=X_test.dtype)
        Y_cur = np.zeros((args.batch_size,) + Y_test.shape[1:],
                         dtype=Y_test.dtype)

        for batch in range(nb_batches):

            # Must not use the `batch_indices` function here, because it
            # repeats some examples.
            # It's acceptable to repeat during training, but not eval.
            start = batch * args.batch_size
            end = min(len(X_test), start + args.batch_size)

            # The last batch may be smaller than all others. This should not
            # affect the accuarcy disproportionately.
            cur_batch_size = end - start
            X_cur[:cur_batch_size] = X_test[start:end]
            Y_cur[:cur_batch_size] = Y_test[start:end]
            feed_dict = {x: X_cur, y: Y_cur}
            if feed is not None:
                feed_dict.update(feed)
            handle = sess.partial_run_setup([adversarial_image, pred_adv_x, pred_not_equal_orig], feeds=[x, y])
            adv_image_np = sess.partial_run(handle, adversarial_image, feed_dict=feed_dict)
            adv_pred_np = sess.partial_run(handle, pred_adv_x)
            cur_not_equal_preds = sess.partial_run(handle, pred_not_equal_orig)
            # print("attack success rate is {}".format(cur_not_equal_preds.mean()))
            adv_images_total.extend(adv_image_np)
            adv_pred_total.extend(adv_pred_np)
            gt_label_total.extend(np.argmax(Y_cur, axis=-1))

            success_rate += cur_not_equal_preds[:cur_batch_size].sum()

        # Divide by number of examples to get final value
        success_rate /= len(X_test)
        adv_images_total = np.stack(adv_images_total)
        adv_pred_total = np.stack(adv_pred_total)
        gt_label_total = np.stack(gt_label_total)

    return adv_images_total, adv_pred_total, gt_label_total, success_rate



def generate_CIFAR10_adv(attacker_name, train_start=0, train_end=60000, test_start=0,
                         test_end=10000, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                         learning_rate=LEARNING_RATE,
                         clean_train=CLEAN_TRAIN,
                         testing=False,
                         nb_filters=NB_FILTERS, num_threads=None,
                         label_smoothing=0.1, args=FLAGS):
    """
    CIFAR10 cleverhans tutorial
    :param attacker_name:
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

    if "batch_size" in ATTACK_PARAM[attacker_name]:
        global BATCH_SIZE
        batch_size = ATTACK_PARAM[attacker_name]["batch_size"]
        BATCH_SIZE = batch_size

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

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
    data = CIFAR(train_start=train_start, train_end=train_end,
                 test_start=test_start, test_end=test_end)
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

    if clean_train:
        model = ModelAllConvolutional('model1', nb_classes, nb_filters,
                                      input_shape=[32, 32, 3])
        preds = model.get_logits(x)  # tf.tensor

        def evaluate():
            do_eval(preds, x_test, y_test, 'clean_train_clean_eval', False)

        resume_files = os.listdir(args.resume)
        loss = CrossEntropy(model, smoothing=label_smoothing)
        if len(resume_files) == 0:
            saver = tf.train.Saver()
            train(sess, loss, None, None,
                  dataset_train=dataset_train, dataset_size=dataset_size,
                  evaluate=evaluate, args=train_params, rng=rng,
                  var_list=model.get_params())  # 训练nb_epochs个epochs
            save_path = saver.save(sess, "{}/model".format(args.resume), global_step=nb_epochs)
            print("Model saved in path: %s" % save_path)
        else:
            # resume from old
            latest_checkpoint = tf.train.latest_checkpoint(args.resume)
            saver = tf.train.Saver()
            saver.restore(sess, latest_checkpoint)

        # Calculate training error
        if testing:
            evaluate()

        # Initialize the Fast Gradient Sign Method (FGSM) attack object and
        # graph
        attacker = ATTACKERS[attacker_name](model, sess=sess)
        param_dict = ATTACK_PARAM[attacker_name]
        print("begin generate adversarial examples of CIFAR-10 using attacker: {}".format(attacker_name))
        adv_x = attacker.generate(x, **param_dict)  # tensor
        preds_adv = model.get_logits(adv_x)
        # generate adversarial examples

        adv_images_total, adv_pred_total, gt_label_total, success_rate = do_generate_eval(adv_x, preds_adv, x_train,
                                                                                          y_train,
                                                                                          "clean_train_adv_eval", True)
        print("attacker: {} attack successful rate for CIFAR-10 train dataset is {}".format(attacker_name, success_rate))
        adv_images_total, adv_pred_total, gt_label_total, success_rate = do_generate_eval(adv_x, preds_adv, x_test,
                                                                                          y_test, "clean_test_adv_eval",
                                                                                          True)
        print("attacker: {} attack successful rate for CIFAR-10 test dataset is {}".format(attacker_name, success_rate))

    return report


def main(argv=None):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    global BATCH_SIZE
    if FLAGS.batch_size != BATCH_SIZE:
        BATCH_SIZE = FLAGS.batch_size
    for attacker_name in ATTACKERS.keys():
        generate_CIFAR10_adv(attacker_name=attacker_name, nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                             learning_rate=FLAGS.learning_rate,
                             clean_train=FLAGS.clean_train,
                             nb_filters=FLAGS.nb_filters, testing=False, args=FLAGS)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_filters', NB_FILTERS,
                         'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                         'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', BATCH_SIZE,
                         'Size of training batches')
    flags.DEFINE_float('learning_rate', LEARNING_RATE,
                       'Learning rate for training')
    flags.DEFINE_integer('gpu', 0,
                         'GPU for training')
    flags.DEFINE_bool('clean_train', CLEAN_TRAIN, 'Train on clean examples')
    flags.DEFINE_string("resume", CIFAR_MODEL_STORE_PATH, 'store model path')
    tf.app.run()
