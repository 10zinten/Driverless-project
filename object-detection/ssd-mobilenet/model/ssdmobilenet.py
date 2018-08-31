import os

import numpy as np
import tensorflow as tf

from model.base_network import MobileNetBase
from model.layers import conv2d, __variable_with_weight_decay


def create_detector(x, depth, mapsize, name, l2_strength):
    with tf.variable_scope(name):
        strides = [1, 1, 1, 1]
        kernel_shape = [3, 3, x.get_shape()[3], depth]
        initializer = tf.contrib.layers.xavier_initializer()

        w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
        b = tf.Variable(tf.zeros(depth), name='biases')

        x = tf.nn.conv2d(x, w, strides=strides, padding='SAME')
        x = tf.nn.bias_add(x, b)
        x = tf.reshape(x, [-1, mapsize.w*mapsize.h, depth])
    return x


class SSDMobileNet:

    def __init__(self, sess, args, preset):
        self.sess = sess
        self.preset = preset
        self.args = args

        self.X = None
        self.labels = None
        self.num_classes = 2 + 1 # 2 for orange and green cone, 1 for gb
        self.num_vars = self.num_classes + 4

        self.classifier = None
        self.locator = None
        self.result = None

        self.loss = None
        self.localization_loss = None
        self.confidence_loss = None
        self.losses = None

        self.base = None

        self.__build()

    def __build(self):
        self.__init_input()

        self.__build_from_mobilenet()
        print("[INFO] Mobilenet graph build successful... ok")

        self.__load_mobilenet()
        print("[INFO] Mobilenet pretrained parameters load successful... ok")

        self.__build_ssd_layers()
        print("[INFO] SDD layers build successful... ok")

        self.__select_feature_maps()
        print("[INFO] Feature maps selecttion successful... ok")

        self.__build_multibox_head()
        print("[INFO] Multibox head build successful... ok")

    def __init_input(self):
         with tf.variable_scope('input'):
            # Input images
            self.X = tf.placeholder(tf.float32,
                                   [self.args.batch_size, self.args.img_size,
                                    self.args.img_size, 3]
                                   )
            self.is_training = tf.placeholder(tf.bool)

    def __build_from_mobilenet(self):
        """ Build the model from MobileNet. """
        self.base = MobileNetBase(self.args, self.X, self.is_training)

    def __load_mobilenet(self):
        self.base.load_pretrained_weights(self.sess)

    def __build_ssd_layers(self):
        with tf.variable_scope('ssd_layer'):
            self.ssd_conv7_1 = conv2d('sdd_conv7_1', self.base.conv6_2_pw,
                                    num_filters=256, kernel_size=(1, 1),
                                    padding='SAME', stride=(1, 1), activation=tf.nn.relu,
                                    batchnorm_enabled=self.args.batchnorm_enabled,
                                    l2_strength=self.args.l2_strength,
                                    is_training=self.is_training, bias=self.args.bias)

            self.ssd_conv7_2 = conv2d('sdd_conv7_2', self.ssd_conv7_1,
                                    num_filters=512, kernel_size=(3, 3),
                                    padding='SAME', stride=(2, 2), activation=tf.nn.relu,
                                    batchnorm_enabled=self.args.batchnorm_enabled,
                                    l2_strength=self.args.l2_strength,
                                    is_training=self.is_training, bias=self.args.bias)

            self.ssd_conv8_1 = conv2d('sdd_conv8_1', self.ssd_conv7_2,
                                    num_filters=128, kernel_size=(1, 1),
                                    padding='SAME', stride=(1, 1), activation=tf.nn.relu,
                                    batchnorm_enabled=self.args.batchnorm_enabled,
                                    l2_strength=self.args.l2_strength,
                                    is_training=self.is_training, bias=self.args.bias)

            self.ssd_conv8_2 = conv2d('sdd_conv8_2', self.ssd_conv8_1,
                                    num_filters=256, kernel_size=(3, 3),
                                    padding='VALID', stride=(2, 2), activation=tf.nn.relu,
                                    batchnorm_enabled=self.args.batchnorm_enabled,
                                    l2_strength=self.args.l2_strength,
                                    is_training=self.is_training, bias=self.args.bias)


    def __select_feature_maps(self):
        self.__maps = [
            # self.base.conv3_1_pw,
            # self.base.conv4_1_pw,
            self.base.conv5_2_pw,
            self.base.conv6_2_pw,
            self.ssd_conv7_2,
            self.ssd_conv8_2
        ]

        # senity check
        for feat in self.__maps:
            print(feat.name, feat.get_shape().as_list())

    def get_maps(self):
        return self.__maps


    def build_optimizer(self):
        """
        Define Loss function for SSD
        Create Optimizer
        """

    def __build_multibox_head(self):
        with tf.variable_scope('multibox_head'):
            self.__detectors = []
            for i in range(len(self.__maps)):
                fmap = self.__maps[i]
                map_size = self.preset.maps[i].size
                for j in range(2+len(self.preset.maps[i].aspect_ratios)):
                    name = 'detector{}_{}'.format(i, j)
                    detector = create_detector(fmap, self.num_vars, map_size, name, self.args.l2_strength)
                    self.__detectors.append(detector)

        with tf.variable_scope('output'):
            output = tf.concat(self.__detectors, axis=1, name='output')
            self.logits = output [:, :, :self.num_classes]

        with tf.variable_scope('result'):
            self.classifier = tf.nn.softmax(self.logits)
            self.locator = output[:, :, self.num_classes:]
            self.result = tf.concat([self.classifier, self.locator],
                                    axis=-1, name='result')


    def __build_names(self):
        '''Name of the feature maps.'''
        pass
