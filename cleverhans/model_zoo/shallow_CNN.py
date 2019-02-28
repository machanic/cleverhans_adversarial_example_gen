import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers as tf_layers


from cleverhans.model import Model
from cleverhans import initializers
import math
from cleverhans.compat import flags
FLAGS = flags.FLAGS

class Shallow10ConvLayersConv(Model):
    def __init__(self, scope, nb_classes, nb_filters, input_shape, **kwargs):
        del kwargs
        Model.__init__(self, scope, nb_classes, locals())
        self.nb_filters = nb_filters
        self.input_shape = input_shape # [32, 32, 3]
        self.dummpy_input = tf.placeholder(tf.float32, [FLAGS.batch_size] + input_shape)

        # Do a dummy run of fprop to create the variables from the start
        # BATCH_SIZE, img_rows, img_cols, nchannels
        self.fprop(self.dummpy_input)
        # Put a reference to the params in self so that the params get pickled
        self.params = self.get_params()

    def fprop(self, x, **kwargs):
        del kwargs  # 这句很重要一定要加
        conv_args = dict(
            activation=tf.nn.leaky_relu,
            kernel_initializer=initializers.HeReLuNormalInitializer,
            kernel_size=3,
            padding='same')
        y = x

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            log_resolution = int(round(
                math.log(self.input_shape[0]) / math.log(2)))
            for scale in range(log_resolution - 2):
                y = tf.layers.conv2d(y, self.nb_filters << scale, **conv_args)
                y = tf.layers.conv2d(y, self.nb_filters << (scale + 1), **conv_args)
                y = tf.layers.average_pooling2d(y, 2, 2)
            y = tf.layers.conv2d(y, self.nb_classes, **conv_args)
            logits = tf.reduce_mean(y, [1, 2])
            return {self.O_LOGITS: logits,
                    self.O_PROBS: tf.nn.softmax(logits=logits)}



class Shallow4ConvLayersConv(Model):

    def __init__(self, scope, img_size, nb_classes, in_channels=3, dim_hidden=64, **kwargs):
        del kwargs
        Model.__init__(self, scope, nb_classes, locals())
        self.img_size = img_size
        self.dim_output = nb_classes
        self.dim_hidden = dim_hidden
        self.in_channels = in_channels
        self.max_pool = True
        self.is_training = False
        # self.weight = self.construct_conv_weights(dim_output=nb_classes, channels=in_channels, dim_hidden=dim_hidden)
        # Do a dummy run of fprop to create the variables from the start
        self.dummpy_input = tf.placeholder(tf.float32, [FLAGS.batch_size,img_size, img_size, in_channels])
        self.fprop(self.dummpy_input)
        self.params = self.get_params()

    def fprop(self, inp, **kwargs):
        del kwargs
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            logits = self.forward_conv(inp, scope=self.scope)
            probs = tf.nn.softmax(logits=logits)
            return {self.O_LOGITS: logits, self.O_PROBS: probs}

    def conv_block(self, inp, reuse, scope, activation=tf.nn.relu, max_pool_pad='valid', max_pool=False):
        """ Perform, conv, batch norm, nonlinearity, and max pool """
        stride, no_stride = [2,2], [1,1]
        dtype = tf.float32
        if max_pool:
            conv_output = tf.layers.conv2d(inp, self.dim_hidden, 3, strides=no_stride,padding="same")
            # conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
        else:
            conv_output = tf.layers.conv2d(inp, self.dim_hidden, 3, strides=stride, padding="same")
            # conv_output = tf.nn.conv2d(inp, cweight, stride, 'SAME') + bweight
        normed = self.normalize(conv_output, activation, reuse, scope, "batch_norm")
        if max_pool:
            normed = tf.layers.max_pooling2d(normed,stride,stride,max_pool_pad)
            # normed = tf.nn.max_pool(normed, stride, stride, max_pool_pad)
        return normed

    def normalize(self, inp, activation, reuse, scope, norm):
        if norm == 'batch_norm':
            return tf.layers.batch_normalization(inp,training=self.is_training)
            # return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
        # elif norm == 'layer_norm':
        #     return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
        elif norm == 'None':
            return activation(inp)


    # def construct_conv_weights(self, dim_output, channels=3, dim_hidden=64):
    #     weights = {}
    #
    #     dtype = tf.float32
    #     conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
    #     fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
    #     k = 3
    #
    #     weights['conv1'] = tf.get_variable('conv1', [k, k, channels, dim_hidden], initializer=conv_initializer,
    #                                        dtype=dtype)
    #     weights['b1'] = tf.Variable(tf.zeros([dim_hidden]))
    #     weights['conv2'] = tf.get_variable('conv2', [k, k, dim_hidden, dim_hidden], initializer=conv_initializer,
    #                                        dtype=dtype)
    #     weights['b2'] = tf.Variable(tf.zeros([dim_hidden]))
    #     weights['conv3'] = tf.get_variable('conv3', [k, k, dim_hidden, dim_hidden], initializer=conv_initializer,
    #                                        dtype=dtype)
    #     weights['b3'] = tf.Variable(tf.zeros([dim_hidden]))
    #     weights['conv4'] = tf.get_variable('conv4', [k, k, dim_hidden, dim_hidden], initializer=conv_initializer,
    #                                        dtype=dtype)
    #     weights['b4'] = tf.Variable(tf.zeros([dim_hidden]))
    #     # assume max pooling
    #     weights['w5'] = tf.Variable(tf.random_normal([dim_hidden, dim_output]), name='w5')
    #     weights['b5'] = tf.Variable(tf.zeros([dim_output]), name='b5')
    #     return weights


    def forward_conv(self, inp, reuse=False, scope=''):
        channels = 3
        # inp = tf.reshape(inp, [-1, img_size, img_size, channels])
        hidden1 = self.conv_block(inp,  reuse, scope + '0', max_pool=self.max_pool)
        hidden2 = self.conv_block(hidden1, reuse, scope + '1',max_pool=self.max_pool)
        hidden3 = self.conv_block(hidden2,  reuse, scope + '2', max_pool=self.max_pool)
        hidden4 = self.conv_block(hidden3,  reuse, scope + '3', max_pool=self.max_pool)
        hidden4 = tf.reduce_mean(hidden4, [1, 2])
        output = tf_layers.fully_connected(hidden4,self.dim_output,None)
        # return tf.matmul(hidden4, weights['w5']) + weights['b5']
        return output


