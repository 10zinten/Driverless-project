import os

import numpy as np
import tensorflow as tf

from model.base_network import MobileNetBase
from model.layers import conv2d

class SSDMobileNet:

    def __init__(self, sess, args, preset):
        self.sess = sess
        self.preset = preset
        self.args = args
        self.image_input = None
        self.result = None
        self.loss = None
        self.localization_loss = None
        self.confidence_loss = None
        self.labels = None
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


    def build_optimizer(self):
        """
        Define Loss function for SSD
        Create Optimizer
        """

    def __build_classifier(self):
        with tf.variable_scope('classifier'):
            self.__classifiers = []
            for i in range(len(self.__maps)):
                fmap = self.__maps[i]
                map_size = self.preset.maps[i].size
                for j in range(2+len(self.preset.maps[i].aspect_ratios)):
                    pass

    def __build_names(self):
        '''Name of the feature maps.'''
        pass
