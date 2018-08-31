import os
import sys
import traceback

import cv2
import tensorflow as tf
import numpy as np

from model.ssdmobilenet import SSDMobileNet
from utils import parse_args
from ssdutils import get_preset_by_name

config_args = parse_args()
sess = tf.Session()
preset = get_preset_by_name('mobilenet160')
ssd = SSDMobileNet(sess, config_args, preset)

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)

def test_frame(f):
    print("\n#####################################################################")
    print("[Testing]: {}".format(' '.join(f.__name__.split('_')[1:])))
    try:
        f()
        print("[Test Status]: Success ...")
    except AssertionError as a:
        print("[Test status]: Fail ...")
        print("[INFO]: ", a.args[0])

def sample_image():
    img = cv2.imread('../prototype/MobileNet/data/test_images/0.jpg')
    print(" - Image before resize: ", img.shape)
    img = cv2.resize(img, (160, 160))
    img = np.expand_dims(img, axis=0)
    print(' - Image reshape: ', img.shape)

    return img

################################################################################
#                                 TEST CASES                                   #
################################################################################

# test the base network feedforward
def test_basenetwork_feedforward():
    img = sample_image()
    out = sess.run([ssd.base.conv6_2_pw], feed_dict={ssd.X: img, ssd.is_training: False})
    print("last conv: ", out[0].shape)

    assert out[0].shape == (1, 5, 5, 1024), "base network out shape not matched"

# test the feature maps shape
def test_feature_map_shape():
    expected_shapes = [
        [1, 10, 10, 512],
        [1, 5, 5, 1024],
        [1, 3, 3, 512],
        [1, 1, 1, 256]
    ]
    fmaps = ssd.get_maps()
    for i, fmap in enumerate(fmaps):
        assert fmap.get_shape().as_list() == expected_shapes[i], "Shape not matching for {}".format(fmap.name)

def test_ssd_confidence_loss():
    img = sample_image()
    labels = np.random.rand(1, 790, 7)
    ssd.build_optimizer()
    feed_dict = {
            ssd.X: img,
            ssd.labels: labels,
            ssd.is_training: False
    }

    conf_loss = sess.run(ssd.confidence_loss, feed_dict=feed_dict)
    print(" - Confidence loss: ", conf_loss)


if __name__ == "__main__":

    test_frame(test_basenetwork_feedforward)
    test_frame(test_feature_map_shape)
    test_frame(test_ssd_confidence_loss)


