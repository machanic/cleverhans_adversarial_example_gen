"""Semantic adversarial examples
"""

from cleverhans.attacks.attack import Attack


class Semantic(Attack):
    """
    Semantic adversarial examples

    https://arxiv.org/abs/1703.06857

    Note: data must either be centered (so that the negative image can be
    made by simple negation) or must be in the interval [-1, 1]

    :param model: cleverhans.model.Model
    :param center: bool
      If True, assumes data has 0 mean so the negative image is just negation.
      If False, assumes data is in the interval [0, max_val]
    :param max_val: float
      Maximum value allowed in the input data
    :param sess: optional tf.Session
    :param dtypestr: dtype of data
    :param kwargs: passed through to the super constructor
    """

    def __init__(self, model, sess, dtypestr='float32', **kwargs):
        super(Semantic, self).__init__(model, sess, dtypestr, **kwargs)
        self.feedable_kwargs = ('center', 'max_val')


    def parse_params(self,
                     center=True,
                     max_val=2.7,
                     **kwargs):
        """
        :param center: If True, assumes data has 0 mean so the negative image is just negation. If False, assumes data is in the interval [0, max_val]
        :param max_val:
        """
        self.center = center
        self.max_val = max_val
        if hasattr(self.model, 'dataset_factory'):
            if 'center' in self.model.dataset_factory.kwargs:
                assert center == self.model.dataset_factory.kwargs['center']
        return True

    def generate(self, x, **kwargs):
        assert self.parse_params(**kwargs)
        if self.center:
            return -x
        return self.max_val - x
