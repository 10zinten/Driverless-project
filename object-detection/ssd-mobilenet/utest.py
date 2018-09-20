import os
import sys
import traceback

import cv2
import tensorflow as tf
import numpy as np

from model.ssdmobilenet import SSDMobileNet
from model.utils import Params
from model.utils import get_filenames_and_labels
from model.input_fn import input_fn
from model.ssdutils import get_preset_by_name, create_labels


# Test set up
json_path = os.path.join('experiments/base_model', 'params.json')
params = Params(json_path)

# Create the input data pipeline
print("Creating the datasets...")
data_dir = 'dataset/cone/'
model_dir = 'experiments/base_model/'
image_dir = os.path.join(data_dir, 'Images')
label_dir = os.path.join(data_dir, 'Labels')

# get the filenames from the train and dev set
demo_filenames, demo_labels = get_filenames_and_labels(image_dir, label_dir, 'train')
# create ssd labels
params.demo_size = len(demo_filenames)

preset = get_preset_by_name('ssdmobilenet160')
demo_labels = create_labels(preset, params.demo_size, 2, demo_labels)
print("[INFO] Demo labels Shape:", demo_labels.shape)
# Create the two iterators over the two datasets
demo_inputs = input_fn(True, demo_filenames, demo_labels, params)
iterator_init_op = demo_inputs['iterator_init_op']

sess = tf.Session()
sess.run(iterator_init_op)
print("Test image shape", sess.run(demo_inputs['images']).shape)
print("Test label shape", sess.run(demo_inputs['labels']).shape)
preset = get_preset_by_name('ssdmobilenet160')
ssd = SSDMobileNet(True, demo_inputs, preset, params)
loss = ssd.losses['total']
optimizer = tf.train.MomentumOptimizer(learning_rate=params.learning_rate,
                                       momentum=params.momentum)
train_op = optimizer.minimize(loss)
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

sess.run([init, iterator_init_op])


def test_frame(f):
    print("\n#####################################################################")
    print("[Test Case]: {}".format(' '.join(f.__name__.split('_')[1:])))
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
    out = sess.run(ssd.base.conv6_2_pw)
    print("last conv: ", out.shape)

    assert out.shape == (params.batch_size, 5, 5, 1024), "base network out shape not matched"

# test the feature maps shape
def test_feature_map_shape():
    expected_shapes = [
        [None, 10, 10, 512],
        [None, 5, 5, 1024],
        [None, 3, 3, 512],
        [None, 1, 1, 256]
    ]
    fmaps = ssd.get_maps()
    for i, fmap in enumerate(fmaps):
        print(" -", fmap.get_shape().as_list(), expected_shapes[i])
        assert fmap.get_shape().as_list() == expected_shapes[i], "Shape not matching for {}".format(fmap.name)

def test_ssd_confidence_loss():
    conf_loss = sess.run(ssd.confidence_loss) #, feed_dict=feed_dict)
    print(" - Confidence loss:", conf_loss)

    assert conf_loss >= 0.0, "Not expected Confidence loss"

def test_ssd_localization_loss():
    # feed_dict = sample_single_datapoint()
    loc_loss = sess.run(ssd.localization_loss) #, feed_dict=feed_dict)
    print(" - Localization loss:", loc_loss)

    assert loc_loss >= 0.0, "Not expected Localization loss"

def test_final_loss():
    # feed_dict = sample_single_datapoint()
    data_loss, reg_loss, loss = sess.run([ssd.data_loss, ssd.reg_loss, ssd.loss])
                                         # feed_dict=feed_dict)
    print(' - Data loss:', data_loss)
    print(' - Regularization loss:', reg_loss)
    print(' - Final loss:', loss)

    assert loss == data_loss+reg_loss, "Loss did not match"

def test_ssd_optimizer():

    # feed_dict = sample_single_datapoint()
    sess.run(iterator_init_op)
    loss_i = sess.run(loss) #, feed_dict=feed_dict)
    print(" - Initial loss:", loss_i)

    for _ in range(10):
        _ = sess.run(train_op) #, feed_dict=feed_dict)

    loss_o = sess.run(loss) #, feed_dict=feed_dict)
    print(" - Optimized loss:", loss_o)

    assert loss_i != loss_o, "Optimizer is not updating the weights"

def test_ssd_label_create(n_samples=3):

    # Create sysnthetic gt_box
    np.random.seed(seed=40)
    gt = []
    for _ in range(n_samples):
        boxes = np.random.rand(3, 4)
        cls = [0, 1, 0]
        gt.append(list(zip(boxes, cls)))

    labels = create_labels(preset, n_samples, 2, gt)

    print(" - Number of samples:", n_samples)
    print(" - Labels shape:", labels.shape)

    assert labels.shape == (n_samples, 790, 7)

if __name__ == "__main__":

    test_frame(test_basenetwork_feedforward)
    test_frame(test_feature_map_shape)
    test_frame(test_ssd_confidence_loss)
    test_frame(test_ssd_localization_loss)
    test_frame(test_final_loss)
    test_frame(test_ssd_optimizer)
    test_frame(test_ssd_label_create)
