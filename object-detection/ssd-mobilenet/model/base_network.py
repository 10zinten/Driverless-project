import tensorflow as tf
from layers import depthwise_separable_conv2d, conv2d, dropout, zero_pad
import os
from utils import load_obj, save_obj
import numpy as np


class MobileNetBase:
    """
    MobileNet Base Class
    """

    def __init__(self,
                 args):

        # init parameters and input
        self.X = None
        self.is_training = None
        self.mean_img = None
        self.nodes = {}
        self.args = args
        self.last_conv = None

        self.pretrained_path = os.path.realpath(self.args.pretrained_path)

        self.__build()

    def __init_input(self):
        with tf.variable_scope('input'):
            # Input images
            self.X = tf.placeholder(tf.float32,
                                    [None, 160, 160, 3])
            self.is_training = tf.placeholder(tf.bool)

    def __init_mean(self):
        # Preparing the mean image.
        img_mean = np.ones((1, 160, 160, 3))
        img_mean[:, :, :, 0] *= 103.939
        img_mean[:, :, :, 1] *= 116.779
        img_mean[:, :, :, 2] *= 123.68
        self.mean_img = tf.constant(img_mean, dtype=tf.float32)

    def __build(self):
        self.__init_mean()
        self.__init_input()
        print("[INFO] creating mobilenet base")
        self.out = self.__init_network()

    def __init_network(self):
        with tf.variable_scope('mobilenet_base'):
            # Preprocessing as done in the paper
            with tf.name_scope('pre_processing'):
                preprocessed_input = (self.X - self.mean_img) / 255.0

            # Model is here!
            conv1_1 = conv2d('conv_1', zero_pad(preprocessed_input), num_filters=int(round(32 * self.args.width_multiplier)),
                             kernel_size=(3, 3),
                             padding='VALID', stride=(2, 2), activation=tf.nn.relu6,
                             batchnorm_enabled=self.args.batchnorm_enabled,
                             is_training=self.is_training, l2_strength=self.args.l2_strength, bias=self.args.bias)
            self.__add_to_nodes([conv1_1])
            ############################################################################################
            conv2_1_dw, conv2_1_pw = depthwise_separable_conv2d('conv_ds_2', zero_pad(conv1_1),
                                                                width_multiplier=self.args.width_multiplier,
                                                                num_filters=64, kernel_size=(3, 3), padding='VALID',
                                                                stride=(1, 1),
                                                                batchnorm_enabled=self.args.batchnorm_enabled,
                                                                activation=tf.nn.relu6,
                                                                is_training=self.is_training,
                                                                l2_strength=self.args.l2_strength,
                                                                biases=(self.args.bias, self.args.bias))
            self.__add_to_nodes([conv2_1_dw, conv2_1_pw])

            conv2_2_dw, conv2_2_pw = depthwise_separable_conv2d('conv_ds_3', zero_pad(conv2_1_pw),
                                                                width_multiplier=self.args.width_multiplier,
                                                                num_filters=128, kernel_size=(3, 3), padding='VALID',
                                                                stride=(2, 2),
                                                                batchnorm_enabled=self.args.batchnorm_enabled,
                                                                activation=tf.nn.relu6,
                                                                is_training=self.is_training,
                                                                l2_strength=self.args.l2_strength,
                                                                biases=(self.args.bias, self.args.bias))
            self.__add_to_nodes([conv2_2_dw, conv2_2_pw])
            ############################################################################################
            conv3_1_dw, conv3_1_pw = depthwise_separable_conv2d('conv_ds_4', zero_pad(conv2_2_pw),
                                                                width_multiplier=self.args.width_multiplier,
                                                                num_filters=128, kernel_size=(3, 3), padding='VALID',
                                                                stride=(1, 1),
                                                                batchnorm_enabled=self.args.batchnorm_enabled,
                                                                activation=tf.nn.relu6,
                                                                is_training=self.is_training,
                                                                l2_strength=self.args.l2_strength,
                                                                biases=(self.args.bias, self.args.bias))
            self.__add_to_nodes([conv3_1_dw, conv3_1_pw])

            conv3_2_dw, conv3_2_pw = depthwise_separable_conv2d('conv_ds_5', zero_pad(conv3_1_pw),
                                                                width_multiplier=self.args.width_multiplier,
                                                                num_filters=256, kernel_size=(3, 3), padding='VALID',
                                                                stride=(2, 2),
                                                                batchnorm_enabled=self.args.batchnorm_enabled,
                                                                activation=tf.nn.relu6,
                                                                is_training=self.is_training,
                                                                l2_strength=self.args.l2_strength,
                                                                biases=(self.args.bias, self.args.bias))
            self.__add_to_nodes([conv3_2_dw, conv3_2_pw])
            ############################################################################################
            conv4_1_dw, conv4_1_pw = depthwise_separable_conv2d('conv_ds_6', zero_pad(conv3_2_pw),
                                                                width_multiplier=self.args.width_multiplier,
                                                                num_filters=256, kernel_size=(3, 3), padding='VALID',
                                                                stride=(1, 1),
                                                                batchnorm_enabled=self.args.batchnorm_enabled,
                                                                activation=tf.nn.relu6,
                                                                is_training=self.is_training,
                                                                l2_strength=self.args.l2_strength,
                                                                biases=(self.args.bias, self.args.bias))
            self.__add_to_nodes([conv4_1_dw, conv4_1_pw])

            conv4_2_dw, conv4_2_pw = depthwise_separable_conv2d('conv_ds_7', zero_pad(conv4_1_pw),
                                                                width_multiplier=self.args.width_multiplier,
                                                                num_filters=512, kernel_size=(3, 3), padding='VALID',
                                                                stride=(2, 2),
                                                                batchnorm_enabled=self.args.batchnorm_enabled,
                                                                activation=tf.nn.relu6,
                                                                is_training=self.is_training,
                                                                l2_strength=self.args.l2_strength,
                                                                biases=(self.args.bias, self.args.bias))
            self.__add_to_nodes([conv4_2_dw, conv4_2_pw])
            ############################################################################################
            conv5_1_dw, conv5_1_pw = depthwise_separable_conv2d('conv_ds_8', zero_pad(conv4_2_pw),
                                                                width_multiplier=self.args.width_multiplier,
                                                                num_filters=512, kernel_size=(3, 3), padding='VALID',
                                                                stride=(1, 1),
                                                                batchnorm_enabled=self.args.batchnorm_enabled,
                                                                activation=tf.nn.relu6,
                                                                is_training=self.is_training,
                                                                l2_strength=self.args.l2_strength,
                                                                biases=(self.args.bias, self.args.bias))
            self.__add_to_nodes([conv5_1_dw, conv5_1_pw])

            conv5_2_dw, conv5_2_pw = depthwise_separable_conv2d('conv_ds_9', zero_pad(conv5_1_pw),
                                                                width_multiplier=self.args.width_multiplier,
                                                                num_filters=512, kernel_size=(3, 3), padding='VALID',
                                                                stride=(1, 1),
                                                                batchnorm_enabled=self.args.batchnorm_enabled,
                                                                activation=tf.nn.relu6,
                                                                is_training=self.is_training,
                                                                l2_strength=self.args.l2_strength,
                                                                biases=(self.args.bias, self.args.bias))
            self.__add_to_nodes([conv5_2_dw, conv5_2_pw])
            ############################################################################################
            conv6_1_dw, conv6_1_pw = depthwise_separable_conv2d('conv_ds_10', zero_pad(conv5_2_pw),
                                                                width_multiplier=self.args.width_multiplier,
                                                                num_filters=1024, kernel_size=(3, 3), padding='VALID',
                                                                stride=(2, 2),
                                                                batchnorm_enabled=self.args.batchnorm_enabled,
                                                                activation=tf.nn.relu6,
                                                                is_training=self.is_training,
                                                                l2_strength=self.args.l2_strength,
                                                                biases=(self.args.bias, self.args.bias))

            self.__add_to_nodes([conv6_1_dw, conv6_1_pw])
            conv6_2_dw, conv6_2_pw = depthwise_separable_conv2d('conv_ds_11', zero_pad(conv6_1_pw),
                                                                width_multiplier=self.args.width_multiplier,
                                                                num_filters=1024, kernel_size=(3, 3), padding='VALID',
                                                                stride=(1, 1),
                                                                batchnorm_enabled=self.args.batchnorm_enabled,
                                                                activation=tf.nn.relu6,
                                                                is_training=self.is_training,
                                                                l2_strength=self.args.l2_strength,
                                                                biases=(self.args.bias, self.args.bias))

            self.__add_to_nodes([conv6_2_dw, conv6_2_pw])
            self.last_conv = conv6_2_pw
            print('Model Created successfully')


    def __restore(self, file_name, sess):
        try:
            print("[INFO] Loading ImageNet pretrained weights...")
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mobilenet_base')
            pretrained = load_obj(file_name)
            run_list = []
            for variable in variables:
                for key, value in pretrained.items():
                    if key in variable.name:
                        run_list.append(tf.assign(variable, value))
            sess.run(run_list)
            print("[INFO] ImageNet Pretrained Weights Loaded Initially\n\n")
        except:
            print("[INFO]No pretrained ImageNet weights exist. Skipping...\n\n")

    def load_pretrained_weights(self, sess):
        self.__restore(self.pretrained_path, sess)

    def __add_to_nodes(self, nodes):
        for node in nodes:
            self.nodes[node.name] = node

    def __init_global_epoch(self):
        """
        Create a global epoch tensor to totally save the process of the training
        :return:
        """
        with tf.variable_scope('global_epoch'):
            self.global_epoch_tensor = tf.Variable(-1, trainable=False, name='global_epoch')
            self.global_epoch_input = tf.placeholder('int32', None, name='global_epoch_input')
            self.global_epoch_assign_op = self.global_epoch_tensor.assign(self.global_epoch_input)

    def __init_global_step(self):
        """
        Create a global step variable to be a reference to the number of iterations
        :return:
        """
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)
