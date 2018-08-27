import os

import numpy as np
import tensorflow as tf

class SSDMobileNet:

    def __init__(self, sess, preset):
        self.preset = preset
        self.sess = sess
        self.__build = False
        self.__build_names()

    def build_from_mobilenet(self, num_classes, a_trous=True):
        """ Build the model from MobileNet. """
        pass

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
