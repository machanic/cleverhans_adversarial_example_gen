from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from cleverhans.model import Model
import tensorflow as tf
from cleverhans.compat import flags
FLAGS = flags.FLAGS


resnet_arg_scope = resnet_utils.resnet_arg_scope

class ResNet10(Model):
    def __init__(self, scope, nb_classes, input_shape, **kwargs):
        del kwargs
        Model.__init__(self, scope, nb_classes, locals())
        self.input_shape = input_shape
        self.is_training = True
        # Do a dummy run of fprop to create the variables from the start
        self.dummpy_input = tf.placeholder(tf.float32, [FLAGS.batch_size] + input_shape)
        self.fprop(self.dummpy_input)
        # Put a reference to the params in self so that the params get pickled
        self.params = self.get_params()

    def fprop(self, x, **kwargs):
        del kwargs
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            logits = resnet_v2_10(x, self.nb_classes, is_training=self.is_training)
            logits = tf.reshape(logits, shape=[-1, self.nb_classes])
        return {self.O_LOGITS: logits,
                    self.O_PROBS: tf.nn.softmax(logits=logits)}

class ResNet18(Model):
    def __init__(self, scope, nb_classes, input_shape, **kwargs):
        del kwargs
        Model.__init__(self, scope, nb_classes, locals())
        self.input_shape = input_shape
        self.is_training = True
        # Do a dummy run of fprop to create the variables from the start
        self.dummpy_input = tf.placeholder(tf.float32, [FLAGS.batch_size] + input_shape)
        self.fprop(self.dummpy_input)
        # Put a reference to the params in self so that the params get pickled
        self.params = self.get_params()

    def fprop(self, x, **kwargs):
        del kwargs
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            logits = resnet_v2_18(x, self.nb_classes, is_training=self.is_training)
            logits = tf.reshape(logits, shape=[-1, self.nb_classes])
        return {self.O_LOGITS: logits,
                    self.O_PROBS: tf.nn.softmax(logits=logits)}



class ResNet50(Model):
    def __init__(self, scope, nb_classes, input_shape, **kwargs):
        del kwargs
        Model.__init__(self, scope, nb_classes, locals())
        self.input_shape = input_shape
        self.is_training = True
        # Do a dummy run of fprop to create the variables from the start
        self.dummpy_input = tf.placeholder(tf.float32, [FLAGS.batch_size] + input_shape)
        self.fprop(self.dummpy_input)
        # Put a reference to the params in self so that the params get pickled
        self.params = self.get_params()

    def fprop(self, x, **kwargs):
        del kwargs
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            logits = resnet_v2_50(x, self.nb_classes, is_training=self.is_training)
            logits = tf.reshape(logits, shape=[-1, self.nb_classes])
        return {self.O_LOGITS: logits,
                    self.O_PROBS: tf.nn.softmax(logits=logits)}

@add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate=1,
               outputs_collections=None, is_training=True,
               scope=None):
    """Bottleneck residual unit variant with BN before convolutions.
    This is the full preactivation residual unit variant proposed in [2]. See
    Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
    variant which has an extra bottleneck layer.
    When putting together two consecutive ResNet blocks that use this unit, one
    should use stride = 2 in the last unit of the first block.
    Args:
      inputs: A tensor of size [batch, height, width, channels].
      depth: The depth of the ResNet unit output.
      depth_bottleneck: The depth of the bottleneck layers.
      stride: The ResNet unit's stride. Determines the amount of downsampling of
        the units output compared to its input.
      rate: An integer, rate for atrous convolution.
      outputs_collections: Collection to add the ResNet unit output.
      scope: Optional variable_scope.
    Returns:
      The ResNet unit's output.
    """
    with variable_scope.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = tf.layers.batch_normalization(inputs, training=is_training, reuse=tf.AUTO_REUSE)
        preact = tf.nn.relu(preact)
        # preact = layers.batch_norm(
        #     inputs, activation_fn=nn_ops.relu, scope='preact')
        if depth == depth_in:
            shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = layers_lib.conv2d(
                preact,
                depth, [1, 1],
                stride=stride,
                normalizer_fn=None,
                activation_fn=None,
                scope='shortcut')

        residual = layers_lib.conv2d(
            preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = resnet_utils.conv2d_same(
            residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
        residual = layers_lib.conv2d(
            residual,
            depth, [1, 1],
            stride=1,
            normalizer_fn=None,
            activation_fn=None,
            scope='conv3')

        output = shortcut + residual

        return utils.collect_named_outputs(outputs_collections, sc.name, output)


def resnet_v2(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True):
    """Generator for v2 (preactivation) ResNet models.
    This function generates a family of ResNet v2 models. See the resnet_v2_*()
    methods for specific model instantiations, obtained by selecting different
    block instantiations that produce ResNets of various depths.
    Training for image classification on Imagenet is usually done with [224, 224]
    inputs, resulting in [7, 7] feature maps at the output of the last ResNet
    block for the ResNets defined in [1] that have nominal stride equal to 32.
    However, for dense prediction tasks we advise that one uses inputs with
    spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
    this case the feature maps at the ResNet output will have spatial shape
    [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
    and corners exactly aligned with the input image corners, which greatly
    facilitates alignment of the features to the image. Using as input [225, 225]
    images results in [8, 8] feature maps at the output of the last ResNet block.
    For dense prediction tasks, the ResNet needs to run in fully-convolutional
    (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
    have nominal stride equal to 32 and a good choice in FCN mode is to use
    output_stride=16 in order to increase the density of the computed features at
    small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.
    Args:
      inputs: A tensor of size [batch, height_in, width_in, channels].
      blocks: A list of length equal to the number of ResNet blocks. Each element
        is a resnet_utils.Block object describing the units in the block.
      num_classes: Number of predicted classes for classification tasks. If None
        we return the features before the logit layer.
      is_training: whether batch_norm layers are in training mode.
      global_pool: If True, we perform global average pooling before computing the
        logits. Set to True for image classification, False for dense prediction.
      output_stride: If None, then the output will be computed at the nominal
        network stride. If output_stride is not None, it specifies the requested
        ratio of input to output spatial resolution.
      include_root_block: If True, include the initial convolution followed by
        max-pooling, if False excludes it. If excluded, `inputs` should be the
        results of an activation-less convolution.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
      net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
        If global_pool is False, then height_out and width_out are reduced by a
        factor of output_stride compared to the respective height_in and width_in,
        else both height_out and width_out equal one. If num_classes is None, then
        net is the output of the last ResNet block, potentially after global
        average pooling. If num_classes is not None, net contains the pre-softmax
        activations.
      end_points: A dictionary from components of the network to the corresponding
        activation.
    Raises:
      ValueError: If the target output_stride is not valid.
    """

    net = inputs
    if include_root_block:
        if output_stride is not None:
            if output_stride % 4 != 0:
                raise ValueError('The output_stride needs to be a multiple of 4.')
            output_stride /= 4
        # We do not include batch normalization or activation functions in
        # conv1 because the first ResNet unit will perform these. Cf.
        # Appendix of [2].
        with arg_scope(
                [layers_lib.conv2d], activation_fn=None, normalizer_fn=None):
            net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
        net = layers.max_pool2d(net, [3, 3], stride=2, scope='pool1')
    net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
    # This is needed because the pre-activation variant does not have batch
    # normalization or activation functions in the residual unit output. See
    # Appendix of [2].
    net = tf.layers.batch_normalization(net, training=is_training,reuse=tf.AUTO_REUSE)
    net = tf.nn.relu(net)
    # net = layers.batch_norm(
    #     net, activation_fn=nn_ops.relu, scope='postnorm')
    if global_pool:
        # Global average pooling.
        net = math_ops.reduce_mean(net, [1, 2], name='pool5', keepdims=True)
    if num_classes is not None:
        net = layers_lib.conv2d(
            net,
            num_classes, [1, 1],
            activation_fn=None,
            normalizer_fn=None,
            scope='logits')
        # Convert end_points_collection into a dictionary of end_points.
        return net


resnet_v2.default_image_size = 224


def resnet_v2_block(scope, base_depth, num_units, stride,  is_training):
    """Helper function for creating a resnet_v2 bottleneck block.
    Args:
      scope: The scope of the block.
      base_depth: The depth of the bottleneck layer for each unit.
      num_units: The number of units in the block.
      stride: The stride of the block, implemented as a stride in the last unit.
        All other units have stride=1.
    Returns:
      A resnet_v2 bottleneck block.
    """
    return resnet_utils.Block(scope, bottleneck, [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': 1, "is_training":is_training
    }] * (num_units - 1) + [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': stride, "is_training":is_training
    }])



def resnet_v2_10(inputs,
                 num_classes=None,
                 global_pool=True,
                 output_stride=None,is_training=True):
    blocks = [
        resnet_v2_block('block1', base_depth=64, num_units=1, stride=2, is_training=is_training),
        resnet_v2_block('block2', base_depth=128, num_units=1, stride=2, is_training=is_training),
        resnet_v2_block('block3', base_depth=256, num_units=1, stride=2, is_training=is_training),
        resnet_v2_block('block4', base_depth=512, num_units=1, stride=1, is_training=is_training),
    ]
    return resnet_v2(
        inputs,
        blocks,
        num_classes,
        is_training,
        global_pool,
        output_stride,
        include_root_block=True)

def resnet_v2_18(inputs,
                 num_classes=None,
                 global_pool=True,
                 output_stride=None, is_training=True):
    blocks = [
        resnet_v2_block('block1', base_depth=64, num_units=2, stride=2,  is_training=is_training),
        resnet_v2_block('block2', base_depth=128, num_units=2, stride=2, is_training=is_training),
        resnet_v2_block('block3', base_depth=256, num_units=2, stride=2, is_training=is_training),
        resnet_v2_block('block4', base_depth=512, num_units=2, stride=1, is_training=is_training),
    ]
    return resnet_v2(
        inputs,
        blocks,
        num_classes,
        is_training,
        global_pool,
        output_stride,
        include_root_block=True)

def resnet_v2_50(inputs,
                 num_classes=None,
                 global_pool=True,
                 output_stride=None,is_training=True,):
    """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        resnet_v2_block('block1', base_depth=64, num_units=3, stride=2,is_training=is_training),
        resnet_v2_block('block2', base_depth=128, num_units=4, stride=2,is_training=is_training),
        resnet_v2_block('block3', base_depth=256, num_units=6, stride=2,is_training=is_training),
        resnet_v2_block('block4', base_depth=512, num_units=3, stride=1,is_training=is_training),
    ]
    return resnet_v2(
        inputs,
        blocks,
        num_classes,
        is_training,
        global_pool,
        output_stride,
        include_root_block=True)


def resnet_v2_101(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  reuse=None,
                  scope='resnet_v2_101'):
    """ResNet-101 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        resnet_v2_block('block1', base_depth=64, num_units=3, stride=2,is_training=is_training),
        resnet_v2_block('block2', base_depth=128, num_units=4, stride=2,is_training=is_training),
        resnet_v2_block('block3', base_depth=256, num_units=23, stride=2,is_training=is_training),
        resnet_v2_block('block4', base_depth=512, num_units=3, stride=1,is_training=is_training),
    ]
    return resnet_v2(
        inputs,
        blocks,
        num_classes,
        is_training,
        global_pool,
        output_stride,
        include_root_block=True)


def resnet_v2_152(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  reuse=None,
                  scope='resnet_v2_152'):
    """ResNet-152 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        resnet_v2_block('block1', base_depth=64, num_units=3, stride=2,is_training=is_training),
        resnet_v2_block('block2', base_depth=128, num_units=8, stride=2,is_training=is_training),
        resnet_v2_block('block3', base_depth=256, num_units=36, stride=2,is_training=is_training),
        resnet_v2_block('block4', base_depth=512, num_units=3, stride=1,is_training=is_training),
    ]
    return resnet_v2(
        inputs,
        blocks,
        num_classes,
        is_training,
        global_pool,
        output_stride,
        include_root_block=True)


def resnet_v2_200(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  reuse=None,
                  scope='resnet_v2_200'):
    """ResNet-200 model of [2]. See resnet_v2() for arg and return description."""
    blocks = [
        resnet_v2_block('block1', base_depth=64, num_units=3, stride=2,is_training=is_training),
        resnet_v2_block('block2', base_depth=128, num_units=24, stride=2,is_training=is_training),
        resnet_v2_block('block3', base_depth=256, num_units=36, stride=2,is_training=is_training),
        resnet_v2_block('block4', base_depth=512, num_units=3, stride=1,is_training=is_training),
    ]
    return resnet_v2(
        inputs,
        blocks,
        num_classes,
        is_training,
        global_pool,
        output_stride,
        include_root_block=True,
        reuse=reuse,
        scope=scope)
