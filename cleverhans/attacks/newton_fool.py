# MIT License
#
# Copyright (C) IBM Corporation 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import absolute_import, division, print_function, unicode_literals

import warnings

import numpy as np
import tensorflow as tf

from cleverhans.attacks.attack import Attack


class NewtonFool(Attack):
    """
    translate from
    https://github.com/IBM/adversarial-robustness-toolbox/blob/master/art/attacks/newtonfool.py
    untested yet
    Implementation of the attack from Uyeong Jang et al. (2017). Paper link: http://doi.acm.org/10.1145/3134600.3134635
    """

    def __init__(self, model, sess, dtypestr='float32', **kwargs):
        """
        Create a NewtonFool attack instance.
        :param model: A trained model.
        :param nb_iter: The maximum number of iterations.
        :type nb_iter: `int`
        :param eta: The eta coefficient.
        :type eta: `float`
        :param batch_size: Batch size
        :type batch_size: `int`
        """
        super(NewtonFool, self).__init__(model, sess, dtypestr, **kwargs)
        self.feedable_kwargs = ('nb_iter', 'eta', 'batch_size')


    def parse_params(self,
                     eta=0.01,
                     nb_iter=1000,
                     y=None,
                     y_target=None,
                     clip_min=None,
                     clip_max=None,
                     batch_size=128,
                     **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:

        :param eps: (optional float) attack step size (input variation)
        :param ord: (optional) Order of the norm (mimics NumPy).
                    Possible values: np.inf, 1 or 2.
        :param y: (optional) A tensor with the true labels. Only provide
                  this parameter if you'd like to use true labels when crafting
                  adversarial samples. Otherwise, model predictions are used as
                  labels to avoid the "label leaking" effect (explained in this
                  paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param sanity_checks: bool, if True, include asserts
          (Turn them off to use less runtime / memory or for unit tests that
          intentionally pass strange input)
        """
        # Save attack-specific parameters

        self.eta = eta
        self.y = y
        self.y_target = y_target
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.nb_iter = nb_iter
        self.batch_size = batch_size

        # Check if order of the norm is acceptable given current implementation
        if len(kwargs.keys()) > 0:
            warnings.warn("kwargs is unused and will be removed on or after "
                          "2019-04-26.")
        return True


    def generate(self, x, **kwargs):
        """
        Generate adversarial samples and return them in a Numpy array.
        :param x: An array with the original inputs to be attacked.
        :type x: `tf.tensor`
        :param kwargs: Attack-specific parameters used by child classes.
        :type kwargs: `dict`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        assert self.parse_params(**kwargs)
        x_adv = tf.identity(x)
        pred_class, _nb_classes = self.get_or_guess_labels(x, kwargs)
        _nb_classes = int(_nb_classes)
        # Initialize variables
        batch = x_adv
        norm_batch = tf.norm(tf.reshape(batch, (tf.shape(batch)[0], -1)),  ord=2, axis=1)  # 1-D vector shape = tf.shape(batch)[0]
        l = tf.cast(self.model.get_predicted_class(x),tf.int32)
        l_b = tf.cast(tf.one_hot(l, depth=_nb_classes, on_value=1, off_value=0), tf.bool)  # to_categoricalm 2-D

        def cond(i, _, __, ___, _____):
            return tf.less(i, self.nb_iter)

        def body(i, batch, l, l_b, nb_classes):
            preds_score = self.model.get_probs(batch)
            score = tf.boolean_mask(preds_score, l_b)  # Compute score， get_probs return shape of (nb_inputs, self.nb_classes)
            grads, = tf.gradients(score, batch)  #    shape = input shape
            with tf.control_dependencies([tf.assert_equal(tf.shape(batch), tf.shape(grads))]):
                norm_grad = tf.norm(tf.reshape(grads, [tf.shape(batch)[0], -1]), axis=1, ord=2)
                theta = self._compute_theta(norm_batch, score, norm_grad, nb_classes)
                # Pertubation
                di_batch = NewtonFool._compute_pert(theta, grads, norm_grad)
                batch = tf.add(batch, di_batch)
                batch = tf.stop_gradient(batch)
            return i+1, batch, l, l_b, nb_classes

        _, new_batch, _, _, _ = tf.while_loop(cond, body, (tf.zeros([]), batch, l, l_b, float(_nb_classes)),
                                       back_prop=True, maximum_iterations=self.nb_iter)
        x_adv = tf.clip_by_value(new_batch, self.clip_min, self.clip_max)
        return x_adv

    def _compute_theta(self, norm_batch, score, norm_grad, nb_classes):
        """
        Function to compute the theta at each step.
        :param norm_batch: norm of a batch.
        :type norm_batch: `np.ndarray`, 1-D vector
        :param score: softmax value at the attacked class.
        :type score: `np.ndarray` 1-D vector
        :param norm_grad: norm of gradient values at the attacked class.
        :type norm_grad: `np.ndarray`
        :return: theta value.
        :rtype: `np.ndarray`
        """
        equ1 = self.eta * tf.multiply(norm_batch, norm_grad)
        equ2 = score - 1.0 / nb_classes  # tf 支持
        result = tf.math.minimum(equ1, equ2)
        return result

    @staticmethod
    def _compute_pert(theta, grads, norm_grad):
        """
        Function to compute the pertubation at each step.
        :param theta: theta value at the current step.
        :type theta: `np.ndarray`
        :param grads: gradient values at the attacked class.
        :type grads: `np.ndarray`
        :param norm_grad: norm of gradient values at the attacked class.
        :type norm_grad: `np.ndarray`
        :return: pertubation.
        :rtype: `np.ndarray`
        """
        # Pick a small scalar to avoid division by 0
        tol = 10e-8
        nom = - tf.multiply(tf.expand_dims(tf.expand_dims(tf.expand_dims(theta, axis=-1), axis=-1),axis=-1), grads)
        denom = tf.math.square(norm_grad)
        denom = tf.where(denom < tol, tf.constant(tol,shape=denom.get_shape().as_list()
                                                  ,dtype=tf.float32), denom)  # denom[denom < tol] = tol
        result = tf.div(nom, tf.expand_dims(tf.expand_dims(tf.expand_dims(denom, axis=-1), axis=-1), axis=-1)) # denom[:, None, None, None]

        return result

