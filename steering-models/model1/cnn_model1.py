"""
This CNN model1 is based on ../cnn_architecture.txt.
"""

import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncate_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def new_conv_layer(input,
                   num_input_channels,
                   filter_size,
                   num_filters,
                   use_pooling=True):

    # Create weights and baises
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = weight_variable(shape=shape)
    biases = bias_variable(shape=num_filters)

    layer = tf.nn.conv2d(input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    layer = tf.nn.relu(layer)

    return layer
