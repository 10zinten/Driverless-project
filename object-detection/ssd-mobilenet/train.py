import tensorflow as tf
import numpy as np

from model.ssdmobilenet import SSDMobileNet
from utils import parse_args
from ssdutils import get_preset_by_name


if __name__ == "__main__":
    config_args = parse_args()

    sess = tf.Session()

    preset = get_preset_by_name('mobilenet160')

    ssd = SSDMobileNet(sess, config_args, preset)
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    print("classifier:", ssd.classifier.get_shape())
    print("Locator:", ssd.locator.get_shape())
    print("Result:", ssd.result.get_shape())


