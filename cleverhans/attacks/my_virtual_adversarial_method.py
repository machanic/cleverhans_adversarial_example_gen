"""The VirtualAdversarialMethod attack

"""

import warnings
import tensorflow as tf
from cleverhans.attacks.attack import Attack
from cleverhans.model import Model, CallableModelWrapper
from cleverhans.model import wrapper_warning_logits
from cleverhans.utils_tf import l2_batch_normalize, kl_with_logits
from scipy.stats import entropy

class VirtualAdversarialMethod(Attack):
    """
    This attack was originally proposed by Miyato et al. (2016) and was used
    for virtual adversarial training.
    Paper link: https://arxiv.org/abs/1507.00677

    :param model: cleverhans.model.Model
    :param sess: optional tf.Session
    :param dtypestr: dtype of the data
    :param kwargs: passed through to super constructor
    """

    def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        if not isinstance(model, Model):
            wrapper_warning_logits()
            model = CallableModelWrapper(model, 'logits')

        super(VirtualAdversarialMethod, self).__init__(model, sess, dtypestr,
                                                       **kwargs)

        self.feedable_kwargs = ('eps', 'xi', 'clip_min', 'clip_max')
        self.structural_kwargs = ['num_iterations']

    def entropy_func(self, x, y):
        return entropy(x,y)

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.

        :param x: The model's symbolic inputs.
        :param kwargs: See `parse_params`
        """
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        x_adv = tf.identity(x)
        dims = tf.shape(x)
        preds = self.model.get_logits(x_adv)
        tol = 1e-10

        d = tf.random_normal(dims, dtype=tf.float32)
        for _ in range(self.num_iterations):
            d = l2_batch_normalize(d)
            preds_new = self.model.get_logits(x_adv + d)
            kl_div1 = tf.py_func(self.entropy_func, [preds, preds_new], tf.float32)
            d_new =tf.zeros_like(d, dtype=tf.float32)


        return vatm(
            self.model,
            x,
            self.model.get_logits(x),
            eps=self.eps,
            num_iterations=self.num_iterations,
            xi=self.finite_diff,
            clip_min=self.clip_min,
            clip_max=self.clip_max)

    def parse_params(self,
                     eps=2.0,
                     nb_iter=None,
                     finite_diff=1e-6,
                     clip_min=None,
                     clip_max=None,
                     **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:

        :param eps: (optional float )the epsilon (input variation parameter)
        :param nb_iter: (optional) the number of iterations
          Defaults to 1 if not specified
        :param finite_diff: (optional float) the finite difference parameter
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Save attack-specific parameters
        self.eps = eps
        if nb_iter is None:
            nb_iter = 1
        self.num_iterations = nb_iter
        self.finite_diff = finite_diff
        self.clip_min = clip_min
        self.clip_max = clip_max
        if len(kwargs.keys()) > 0:
            warnings.warn("kwargs is unused and will be removed on or after "
                          "2019-04-26.")
        return True




def vatm_tf(model,
         x,
         logits,
         eps,
         num_iterations=1,
         xi=1e-6,
         clip_min=None,
         clip_max=None,
         scope=None):
    """
    Tensorflow implementation of the perturbation method used for virtual
    adversarial training: https://arxiv.org/abs/1507.00677
    :param model: the model which returns the network unnormalized logits
    :param x: the input placeholder
    :param logits: the model's unnormalized output tensor (the input to
                   the softmax layer)
    :param eps: the epsilon (input variation parameter)
    :param num_iterations: the number of iterations
    :param xi: the finite difference parameter
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :param seed: the seed for random generator
    :return: a tensor for the adversarial example
    """
    with tf.name_scope(scope, "virtual_adversarial_perturbation"):
        d = tf.random_normal(tf.shape(x), dtype=tf_dtype)
        for _ in range(num_iterations):
            d = xi * l2_batch_normalize(d)
            logits_d = model.get_logits(x + d)
            kl = kl_with_logits(logits, logits_d)
            Hd = tf.gradients(kl, d)[0]
            d = tf.stop_gradient(Hd)
        d = eps * l2_batch_normalize(d)
        adv_x = x + d
        if (clip_min is not None) and (clip_max is not None):
            adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)
        return adv_x


def vatm(model,
         x,
         logits,
         eps,
         back='tf',
         num_iterations=1,
         xi=1e-6,
         clip_min=None,
         clip_max=None):
    """
    A wrapper for the perturbation methods used for virtual adversarial
    training : https://arxiv.org/abs/1507.00677
    It calls the right function, depending on the
    user's backend.

    :param model: the model which returns the network unnormalized logits
    :param x: the input placeholder
    :param logits: the model's unnormalized output tensor
    :param eps: the epsilon (input variation parameter)
    :param num_iterations: the number of iterations
    :param xi: the finite difference parameter
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :return: a tensor for the adversarial example

    """
    assert back == 'tf'
    # Compute VATM using TensorFlow
    return vatm_tf(
        model,
        x,
        logits,
        eps,
        num_iterations=num_iterations,
        xi=xi,
        clip_min=clip_min,
        clip_max=clip_max)
