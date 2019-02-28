import tensorflow as tf

from cleverhans.attacks.attack import Attack

class CleanIdentity(Attack):
    def __init__(self, model, sess, dtypestr='float32', **kwargs):
        super(CleanIdentity, self).__init__(model, sess, dtypestr, **kwargs)

    def generate(self, x, **kwargs):
        return tf.identity(x)