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
ssd.build_optimizer(learning_rate=5)

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

def sample_single_datapoint():
    # paramters
    batch_size = 1
    num_classes = 3
    num_anchors = 790
    num_vars = 7

    # Read image
    img = cv2.imread('../prototype/MobileNet/data/test_images/0.jpg')
    img = cv2.resize(img, (160, 160))
    img = np.expand_dims(img, axis=0)

    # Create labels: [batch_size, num_anchors]
    # Confidence Ground Truth
    gt_cl = []
    for i in range(batch_size):
        x = np.eye(num_classes)
        # select row at random choice
        gt_cl.append(x[np.random.choice(x.shape[0], size=num_anchors)].tolist())
    gt_cl = np.array(gt_cl)

    # Localization Ground Truth
    gt_loc = np.random.rand(batch_size, num_anchors, num_vars-num_classes)

    label = np.concatenate((gt_cl, gt_loc), axis=-1)

    print(" - Input shape:", img.shape)
    print(" - label shape:", label.shape)

    feed_dict = {
            ssd.X: img,
            ssd.labels: label,
            ssd.is_training: False
        }

    return feed_dict

################################################################################
#                                 TEST CASES                                   #
################################################################################

# test the base network feedforward
def test_basenetwork_feedforward():
    feed_dict = sample_single_datapoint()
    out = sess.run(ssd.base.conv6_2_pw, feed_dict=feed_dict)
    print("last conv: ", out.shape)

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
    feed_dict = sample_single_datapoint()
    conf_loss = sess.run(ssd.confidence_loss, feed_dict=feed_dict)
    print(" - Confidence loss:", conf_loss)

    assert conf_loss >= 0.0, "Not expected Confidence loss"

def test_ssd_localization_loss():
    feed_dict = sample_single_datapoint()
    loc_loss = sess.run(ssd.localization_loss, feed_dict=feed_dict)
    print(" - Localization loss:", loc_loss)

    assert loc_loss >= 0.0, "Not expected Localization loss"

def test_final_loss():
    feed_dict = sample_single_datapoint()
    data_loss, reg_loss, loss = sess.run([ssd.data_loss, ssd.reg_loss, ssd.loss],
                                         feed_dict=feed_dict)
    print(' - Data loss:', data_loss)
    print(' - Regularization loss:', reg_loss)
    print(' - Final loss:', loss)

    assert loss == data_loss+reg_loss, "Loss did not match"

def test_ssd_optimizer():
    feed_dict = sample_single_datapoint()

    loss_i = sess.run(ssd.loss, feed_dict=feed_dict)
    print(" - Initial loss:", loss_i)

    for _ in range(10):
        _ = sess.run(ssd.optimizer, feed_dict=feed_dict)

    loss_o = sess.run(ssd.loss, feed_dict=feed_dict)
    print(" - Optimized loss:", loss_o)

    assert loss_i != loss_o, "Optimizer is not updating the weights"


if __name__ == "__main__":

    test_frame(test_basenetwork_feedforward)
    test_frame(test_feature_map_shape)
    test_frame(test_ssd_confidence_loss)
    test_frame(test_ssd_localization_loss)
    test_frame(test_final_loss)
    test_frame(test_ssd_optimizer)


