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


def conv2d(x, W, stride):
    return tf.nn.con2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def max_pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
