import os
import sys
import traceback

import cv2
import tensorflow as tf
import numpy as np

from model.ssdmobilenet import SSDMobileNet
from utils import parse_args

config_args = parse_args()
sess = tf.Session()
ssd = SSDMobileNet(sess, config_args, None)
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

# test the base network feedforward
def test_basenetwork_feedforward():
    img = cv2.imread('../prototype/MobileNet/data/test_images/0.jpg')
    print("Image before resize: ", img.shape)
    img = cv2.resize(img, (160, 160))
    img = np.expand_dims(img, axis=0)
    print('Image reshape: ', img.shape)
    out = sess.run([ssd.base.conv6_2_pw], feed_dict={ssd.base.X: img, ssd.base.is_training: False})
    print("last conv: ", out[0].shape)

    assert out[0].shape == (1, 5, 5, 1024), "base network out shape not matched"

if __name__ == "__main__":

    test_frame(test_basenetwork_feedforward)


