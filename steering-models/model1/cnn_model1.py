'''
CNN model is based the architecture metioned in ../cnn_architecture.txt
'''

import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

