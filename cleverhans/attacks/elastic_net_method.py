"""The ElasticNetMethod attack.
"""

import numpy as np
import tensorflow as tf

from cleverhans.attacks.attack import Attack
from cleverhans.model import Model, CallableModelWrapper, wrapper_warning_logits


class ElasticNetMethod(Attack):
    """
    This attack features L1-oriented adversarial examples and includes
    the C&W L2 attack as a special case (when beta is set to 0).
    Adversarial examples attain similar performance to those
    generated by the C&W L2 attack in the white-box case,
    and more importantly, have improved transferability properties
    and complement adversarial training.
    Paper link: https://arxiv.org/abs/1709.04114

    :param model: cleverhans.model.Model
    :param sess: tf.Session
    :param dtypestr: dtype of the data
    :param kwargs: passed through to super constructor
    """

    def __init__(self, model, sess, dtypestr='float32', **kwargs):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        if not isinstance(model, Model):
            wrapper_warning_logits()
            model = CallableModelWrapper(model, 'logits')

        super(ElasticNetMethod, self).__init__(model, sess, dtypestr, **kwargs)

        self.feedable_kwargs = ('y', 'y_target')

        self.structural_kwargs = [
            'beta', 'decision_rule', 'batch_size', 'confidence',
            'targeted', 'learning_rate', 'binary_search_steps',
            'max_iterations', 'abort_early', 'initial_const', 'clip_min',
            'clip_max'
        ]

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.

        :param x: (required) A tensor with the inputs.
        :param kwargs: See `parse_params`
        """
        assert self.sess is not None, \
            'Cannot use `generate` when no `sess` was provided'
        self.parse_params(**kwargs)

        from cleverhans.attacks_tf import ElasticNetMethod as EAD
        labels, nb_classes = self.get_or_guess_labels(x, kwargs)

        attack = EAD(self.sess, self.model, self.beta,
                     self.decision_rule, self.batch_size, self.confidence,
                     'y_target' in kwargs, self.learning_rate,
                     self.binary_search_steps, self.max_iterations,
                     self.abort_early, self.initial_const, self.clip_min,
                     self.clip_max, nb_classes,
                     x.get_shape().as_list()[1:])

        def ead_wrap(x_val, y_val):
            return np.array(attack.attack(x_val, y_val), dtype=self.np_dtype)

        wrap = tf.py_func(ead_wrap, [x, labels], self.tf_dtype)
        wrap.set_shape(x.get_shape())

        return wrap

    def parse_params(self,
                     y=None,
                     y_target=None,
                     beta=1e-2,
                     decision_rule='EN',
                     batch_size=1,
                     confidence=0,
                     learning_rate=1e-2,
                     binary_search_steps=9,
                     max_iterations=1000,
                     abort_early=False,
                     initial_const=1e-3,
                     clip_min=0,
                     clip_max=1):
        """
        :param y: (optional) A tensor with the true labels for an untargeted
                  attack. If None (and y_target is None) then use the
                  original labels the classifier assigns.
        :param y_target: (optional) A tensor with the target labels for a
                  targeted attack.
        :param beta: Trades off L2 distortion with L1 distortion: higher
                     produces examples with lower L1 distortion, at the
                     cost of higher L2 (and typically Linf) distortion
        :param decision_rule: EN or L1. Select final adversarial example from
                              all successful examples based on the least
                              elastic-net or L1 distortion criterion.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the perturbation
                                    and confidence of the classification. Set
                                    'initial_const' to a large value and fix
                                    this param to 1 for speed.
        :param max_iterations: The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early abort when the total
                            loss starts to increase (greatly speeds up attack,
                            but hurts performance, particularly on ImageNet)
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the perturbation
                              and confidence of classification.
                              If binary_search_steps is large, the initial
                              constant is not important. A smaller value of
                              this constant gives lower distortion results.
                              For computational efficiency, fix
                              binary_search_steps to 1 and set this param
                              to a large value.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # ignore the y and y_target argument
        self.beta = beta
        self.decision_rule = decision_rule
        self.batch_size = batch_size
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max
