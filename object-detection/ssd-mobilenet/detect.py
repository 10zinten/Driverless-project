import os
import argparse

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

from model.ssdmobilenet import SSDMobileNet
from model.ssdutils import get_preset_by_name, get_anchors_for_preset
from model.ssdutils import decode_boxes, suppress_overlaps
from model.utils import Params, draw_box

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model/',
                    help='experiment model directory')
parser.add_argument('--image',
                    help='image file name')
parser.add_argument('--restore_from', default='best_weights',
                    help='Subdirectory of model dir file containing the weights')


if __name__ == "__main__":

    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    params = Params(json_path)
    preset = get_preset_by_name('ssdmobilenet160')

    image_string = tf.read_file(args.image)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image = tf.expand_dims(image, 0)

    print(image.get_shape().as_list())

    inputs = {'images': image, 'labels': None}

    # Build the model
    # MODEL: define the layers of the model
    with tf.variable_scope('model'):
        ssd = SSDMobileNet(False, inputs, preset, params)
        result = ssd.result

    # list all the variables of graphs
    # for var in tf.all_variables():
    #     print(var)

    # Initialize the tf.Saver
    saver = tf.train.Saver()

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        # Initialize the lookup table
        sess.run(init)

        # Load weights from the weights subdirectory
        save_path = os.path.join(args.model_dir, args.restore_from)
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, save_path)

        result, image = sess.run([result, image])

    anchors = get_anchors_for_preset(preset)
    boxes = decode_boxes(result[0], anchors, 0.4, None)
    print('Num of decoded boxes:', len(boxes))
    boxes = suppress_overlaps(boxes)
    print('Num of final boxes', len(boxes))

    # Plot the boxes image
    image = np.asarray(Image.open(args.image))
    for box in boxes:
        draw_box(image, box)
    cv2.imwrite('test/result/test.jpeg', image)
