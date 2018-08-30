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
        self.base = None
        self.__build()

    def __build(self):
        self.build_from_mobilenet(2)

    def build_from_mobilenet(self, num_classes):
        """ Build the model from MobileNet. """
        self.num_classes = num_classes + 1
        self.base = MobileNetBase(self.args)
        self.base.load_pretrained_weights(self.sess)


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
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    # test the base network feedforward
    import cv2
    img = cv2.imread('../../prototype/MobileNet/data/test_images/0.jpg')
    print("Image before resize: ", img.shape)
    img = cv2.resize(img, (160, 160))
    img = np.expand_dims(img, axis=0)
    print('Image reshape: ', img.shape)
    last_conv = sess.run([ssd.base.last_conv], feed_dict={ssd.base.X: img, ssd.base.is_training: False})
    print("last conv: ", last_conv[0].shape)
