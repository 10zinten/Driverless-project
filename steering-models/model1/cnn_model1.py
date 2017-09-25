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


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def max_pool2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


class ConvModel1(object):
    def __init__(self, is_training=True):
        x = tf.placeholder(tf.float32, shape=[None, 128*128*3], name='x')
        y_ = tf.placeholder(tf.float32, shape=[None, 1])
        x_image = tf.reshape(x, [-1, 128, 128, 3])

        # Conv layer 1
        self.W_conv1 = weight_variable([3, 3, 1, 32])
        self.b_conv1 = bias_variable([32])
        self.h_conv1 = max_pool2x2(tf.nn.relu(conv2d(x_image, self.W_conv1, 1) + self.b_conv1))

        # Conv layer 2
        self.W_conv2 = weight_variable([3, 3, 32, 64])
        self.b_conv2 = bias_variable([64])
        self.h_conv2 = max_pool2x2(tf.nn.relu(conv2d(self.h_conv1, self.W_conv2, 1) + self.b_conv2))

        # Conv layer 3
        self.W_conv2 = weight_variable([3, 3, 64, 128])
        self.b_conv2 = bias_variable([128])
        self.h_conv2 = max_pool2x2(tf.nn.relu(conv2d(self.h_conv1, self.W_conv2, 1) + self.b_conv2))


        # Fully Connect layer 1
        self.W_fc1 = weight_variable([16*16*128, 1024])
        self.b_fc1 = bias_variable([1024])

        self.h_conv5_flat = tf.reshape(self.h_conv2, [-1, 16*16*128])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv5_flat, self.W_fc1) + self.b_fc1)
        print("Fc1 output shape: ", self.h_fc1)

        # Output layer
        self.W_fc2 = weight_variable([1024, 1])
        self.b_fc2 = bias_variable([1])
        h_fc2 = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2
        print("Final output shape: ", h_fc2)

        self.x = x
        self.y_ = y_
        self.y = h_fc2
