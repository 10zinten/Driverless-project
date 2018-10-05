import os

import numpy as np
import tensorflow as tf

from model.base_network import MobileNetBase
from model.layers import conv2d




def smooth_l1_loss(x):
    square_loss = 0.5 * x**2
    absolute_loss = tf.abs(x)
    return tf.where(tf.less(absolute_loss, 1.), square_loss, absolute_loss-0.5)

class SSDMobileNet:

    def __init__(self, is_training, inputs, preset, args):
        self.preset = preset
        self.args = args
        self.is_training = is_training

        self.X = inputs['images']
        self.labels = inputs['labels']
        self.num_classes = 2 + 1 # 2 for orange and green cone, 1 for gb
        self.num_vars = self.num_classes + 4

        self.classifier = None
        self.locator = None
        self.result = None
        self.optimizer = None

        self.localization_loss = None
        self.confidence_loss = None
        self.data_loss = None
        self.reg_loss = None
        self.loss = None
        self.losses = None

        self.base = None

        self.__build()

    def __build(self):

        self.__build_from_mobilenet()
        print("[INFO] Mobilenet graph build successful... ok")

        # self.__load_mobilenet()
        #  print("[INFO] Mobilenet pretrained parameters load successful... ok")

        self.__build_ssd_layers()
        print("[INFO] SDD layers build successful... ok")

        self.__select_feature_maps()
        print("[INFO] Feature maps selecttion successful... ok")

        self.__build_multibox_head()
        print("[INFO] Multibox head build successful... ok")

        if self.labels is not None:
            self.__build_loss_function()


    def __build_from_mobilenet(self):
        """ Build the model from MobileNet. """
        self.base = MobileNetBase(self.args, self.X, self.is_training)

    def load_mobilenet(self, sess):
        self.base.load_pretrained_weights(sess)

    def __build_ssd_layers(self):
        with tf.variable_scope('ssd_layer'):
            conv6_2_pw_stop = tf.stop_gradient(self.base.conv6_2_pw)

            self.ssd_conv7_1 = conv2d('sdd_conv7_1', conv6_2_pw_stop,
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
            self.base.conv3_1_pw,    # 40
            self.base.conv4_1_pw,    # 20
            self.base.conv5_2_pw,    # 10
            self.base.conv6_2_pw,    # 5
            self.ssd_conv7_2,        # 3
            self.ssd_conv8_2         # 1
        ]


    def get_maps(self):
        return self.__maps

    def __create_detector(self, x, depth, mapsize, name, l2_strength):
        x = conv2d(name, x, num_filters=depth, kernel_size=(3, 3),
                   padding='SAME', stride=(1, 1), activation=tf.nn.relu,
                   batchnorm_enabled=self.args.batchnorm_enabled,
                   l2_strength=self.args.l2_strength,
                   is_training=self.is_training, bias=self.args.bias)

        x = tf.reshape(x, [-1, mapsize.w*mapsize.h, depth])

        return x


    def __build_multibox_head(self):
        with tf.variable_scope('multibox_head'):
            self.__detectors = []
            for i in range(len(self.__maps)):
                fmap = self.__maps[i]
                map_size = self.preset.maps[i].size
                for j in range(2+len(self.preset.maps[i].aspect_ratios)):
                    name = 'detector{}_{}'.format(i, j)
                    detector = self. __create_detector(fmap, self.num_vars, map_size, name, self.args.l2_strength)
                    self.__detectors.append(detector)
        print(" - [INFO] Number of detector: ", len(self.__detectors))
        print(" - [INFO] Fisrt detector shape: ", self.__detectors[0].get_shape().as_list())

        with tf.variable_scope('output'):
            output = tf.concat(self.__detectors, axis=1, name='output')
            self.logits = output [:, :, :self.num_classes]
        print(" - [INFO] Output shape: ", output.get_shape().as_list())

        with tf.variable_scope('result'):
            self.classifier = tf.nn.softmax(self.logits)
            self.locator = output[:, :, self.num_classes:]
            self.result = tf.concat([self.classifier, self.locator],
                                    axis=-1, name='result')

    def __build_loss_function(self):

        with tf.variable_scope('ground_truth'):
            # Classification gt tensor
            # shape: (batch_size, num_anchors, num_classes)
            gt_cl = self.labels[:, :, :self.num_classes]

            # Localization gt tensor
            # shape: (batch_size, num_anchors, 4)
            gt_loc = self.labels[:, :, self.num_classes:]

            # Batch size
            # Shape: scalar
            batch_size = tf.shape(gt_cl)[0]

        with tf.variable_scope('match_counters'):
            # Number of anchors per sample
            # Shape: (batch_size)
            total_num = tf.ones([batch_size], dtype=tf.int64) * \
                        tf.to_int64(self.preset.num_anchors)

            # Number of negative (not-matched) anchors per sample, computed by
            # counting boxes of the background class in each sample.
            # Shape: (batch_size)
            negatives_num = tf.count_nonzero(gt_cl[:, :, -1], axis=1)

            # Number of positive (matched) anchors per sample
            # Shape: (batch_size)
            positives_num = total_num - negatives_num

            # Number of positives per sample that is division-safe
            # Shape: (batch_size)
            positives_num_safe = tf.where(tf.equal(positives_num, 0),
                                          tf.ones([batch_size ])*10e-15,
                                          tf.to_float(positives_num))

        with tf.variable_scope('match_masks'):
            # Boolean tensor determining whether an anchor is a positive
            # Shape: (batch_size, num_anchors)
            positives_mask = tf.equal(gt_cl[:, :, -1], 0)

            # Boolean tensor determining whether an anchor is a negative
            negatives_mask = tf.logical_not(positives_mask)

        with tf.variable_scope('confidence_loss'):
            # Cross-entorpy tensor
            # Shape: (batch_size, num_anchors)
            conf = tf.nn.softmax_cross_entropy_with_logits_v2(labels=gt_cl,
                                                              logits=self.logits)

            # Sum up the loss of all the positive anchors
            # Positives - the loss of neg anchors is zeroed out
            # Shape: (batch_size, num_anchors)
            positives_conf = tf.where(positives_mask, conf, tf.zeros_like(conf))

            # Total loss of positive anchors
            # Shape: (batch_size)
            positives_sum = tf.reduce_sum(positives_conf, axis=-1)

            # Find neg anchors with highest conf loss
            # Negatives - the loss of positive anchor is zeroed out
            # Shape: (batch_size, num_anchors)
            negatives_conf = tf.where(negatives_mask, conf, tf.zeros_like(conf))

            # Top neg - sorted conf loss with highest one first
            # Shape: (batch_size, num_anchors)
            negatives_top = tf.nn.top_k(negatives_conf, self.preset.num_anchors)[0]

            # Find num of negs we want to keep are
            # Max num of negs to keep per sample - keep 3 time as many as pos
            # anchors in the sample
            # Shape: (batch_size)
            negatives_num_max = tf.minimum(negatives_num, 3*positives_num)

            # mask out superfluous negs and compute the sum of the loss
            # Transposed vector of maximum negs per sample
            # Shape: (batch_size, 1)
            negatives_num_max_t = tf.expand_dims(negatives_num_max, 1)

            # Range tensor: [0, 1, 2, ..., num_anchors-1]
            # Shape: (num_anchors)
            rng = tf.range(0, self.preset.num_anchors, 1)

            # Row range, int64, row of a matrix
            # shape: (1, num_anchors)
            range_row = tf.to_int64(tf.expand_dims(rng, 0))

            # mask of max neg
            # shape: (batch_size, num_anchors)
            negatives_max_mask = tf.less(range_row, negatives_num_max_t)


            # Max negs - all posi and superfluous negs are zeroed out.
            # Shape: (batch_sizei, num_anchors)
            negatives_max = tf.where(negatives_max_mask, negatives_top,
                                     tf.zeros_like(negatives_top))

            # Sum of max negs for each sample
            # Shape: (batch_size)
            negatives_max_sum = tf.reduce_sum(negatives_max, axis=-1)

            # Compute confidence loss for each element
            # Total confidence loss for each sample
            # Shape: (batch_size)
            confidence_loss = tf.add(positives_sum, negatives_max_sum)

            # Total confidence loss normalized by the number of positives per
            # sample
            # Shape: (batch_size)
            confidence_loss = tf.where(tf.equal(positives_num, 0),
                                       tf.zeros([batch_size]),
                                       tf.div(confidence_loss,
                                              positives_num_safe))

            # Mean confidence loss for the batch
            # Shape: scalar
            self.confidence_loss = tf.reduce_mean(confidence_loss,
                                                  name='confidence_loss')

        # Compute the localization loss
        with tf.variable_scope('localization_loss'):
            # Element-wise difference btw the predicted localization loss
            # and ground truth
            # Shape: (batch, num_anchors, 4)
            loc_diff = tf.subtract(self.locator, gt_loc)

            # Smooth L1 loss
            # Shape: (batch_size, num_anchors, 4)
            loc_loss = smooth_l1_loss(loc_diff)

            # Sum of localization losses for each anchor
            # Shape: (batch_size, num_anchors)
            loc_loss_sum = tf.reduce_sum(loc_loss, axis=-1)

            # Positive locs - the loss of negative anchors is zeroed out
            # Shape: (batch_size, num_anchors)
            positive_locs = tf.where(positives_mask, loc_loss_sum,
                                     tf.zeros_like(loc_loss_sum))

            # Total loss of positive anchors
            # Shape: (batch_size)
            localization_loss = tf.reduce_sum(positive_locs, axis=-1)

            # Total localization loss normalized by the number of positives per
            # sample
            # Shape: (batch_size)
            localization_loss = tf.where(tf.equal(positives_num, 0),
                                         tf.zeros([batch_size]),
                                         tf.div(confidence_loss,
                                                positives_num_safe))

            # Mean localizationn loss for the batch
            # Shape: scalar
            self.localization_loss = tf.reduce_mean(localization_loss,
                                                    name='sum_losses')

        # Compute total loss
        with tf.variable_scope('total_loss'):
            # Data Loss - Sum of the localization and confidence loss
            # Shape: (batch_size)
            self.data_loss = tf.add(self.confidence_loss,
                                            self.localization_loss,
                                            name='data_loss')

            # Regularization Loss - L2 loss on weight
            # Shape: scalar
            self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            # Final loss
            # Shape: scalar
            self.loss = tf.add(self.data_loss, self.reg_loss, name='loss')

        # Store the tensors
        self.losses = {
            'total': self.loss,
            'localization': self.localization_loss,
            'confidence': self.confidence_loss,
            'regularization': self.reg_loss
        }


    def __build_names(self):
        '''Name of the orignal and new scopes'''
        self.original_scopes = [
            'conv_1', 'conv_ds_2', 'conv_ds_3', 'conv_ds_4', 'conv_ds_5',
            'conv_ds_5', 'conv_ds_6', 'conv_ds_7', 'conv_ds_8', 'conv_ds_9',
            'conv_ds_10', 'conv_ds_11'
        ]

        self.new_scopes = [
            'sdd_conv7_1', 'sdd_conv7_2', 'sdd_conv8_1', 'sdd_conv8_2'
        ]

        for i in range(len(self.preset.maps)):
            for j in range(2+len(self.preset.maps[i].aspect_ratios)):
                self.new_scopes.append('multibox_head/dectector{}_{}'.format(i, j))
