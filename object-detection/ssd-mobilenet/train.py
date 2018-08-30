import tensorflow as tf
import numpy as np

from model.ssdmobilenet import SSDMobileNet
from utils import parse_args


if __name__ == "__main__":
    config_args = parse_args()

    sess = tf.Session()

    ssd = SSDMobileNet(sess, config_args, None)
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
