import os

import numpy as np
import tensorflow as tf

from base_network import MobileNetBase
from utils import parse_args

class SSDMobileNet:

    def __init__(self, sess, args):
        self.sess = sess
        self.args = args
        self.image_input = None
        self.result = None
        self.loss = None
        self.localization_loss = None
        self.confidence_loss = None
        self.labels = None
        self.losses = None
        self.__build()

    def __build(self):
        self.build_from_mobilenet(2)

    def build_from_mobilenet(self, num_classes):
        """ Build the model from MobileNet. """
        self.num_classes = num_classes + 1
        base = MobileNetBase(self.args)
        base.load_pretrained_weights(self.sess)


    def __load_mobilenet(self, dir):
        sess = self.sess
        pass

    def build_optimizer(self):
        """
        Define Loss function for SSD
        Create Optimizer
        """
        pass

    def __build_names(self):
        '''Name of the feature maps.'''
        pass

if __name__ == "__main__":
    config_args = parse_args()

    sess = tf.Session()
    ssd = SSDMobileNet(sess, config_args)
