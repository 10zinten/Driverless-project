import os
import json
from collections import defaultdict

import tensorflow as tf
import numpy as np

from model.ssdmobilenet import SSDMobileNet
from model.input_fn import input_fn
from model.utils import parse_args, get_filenames_and_labels
from model.ssdutils import get_preset_by_name, create_labels


if __name__ == "__main__":
    # set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load the parametes from json file
    config_args = parse_args()

    # Create the input data pipeline
    data_dir = 'dataset/'
    image_dir = os.path.join(data_dir, 'Images')
    label_dir = os.path.join(data_dir, 'Labels')

    train_filenames, train_labels = get_filenames_and_labels(image_dir, label_dir, 'train')
    dev_filenames, dev_labels = get_filenames_and_labels(image_dir, label_dir, 'dev')

    # create ssd labels
    train_size = len(train_filenames)
    dev_size = len(dev_filenames)

    preset = get_preset_by_name('ssdmobilenet160')
    train_labels = create_labels(preset, train_size, 2, train_labels)
    dev_labels = create_labels(preset, dev_size, 2, dev_labels)

    print(train_labels.shape)
    print(dev_labels.shape)

    '''
    ssd = SSDMobileNet(sess, config_args, preset)
    ssd.build_optimizer()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    '''
