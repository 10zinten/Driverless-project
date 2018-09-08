import os
import json
from collections import defaultdict

import tensorflow as tf
import numpy as np

from model.model_fn import model_fn
from model.input_fn import input_fn
from model.training import train_and_evaluate
from model.utils import parse_args
from model.ssdutils import get_preset_by_name, create_labels, abs2prop, Size


def get_filenames_and_labels(image_dir, label_dir, split):
    with open(os.path.join(label_dir, split+'.json'), 'r') as f:
        datapoints = json.load(f)

    img_size = Size(160, 160)
    dps_anno = defaultdict(lambda: [])
    for dp in datapoints:
        filename = os.path.join(image_dir, dp['filename'])
        if dp['annotations']:
            for ann in dp['annotations']:
                cx, cy, w, h = abs2prop(ann['xmin'], ann['xmax'], ann['ymin'], ann['ymax'], img_size)
                bb = np.array([cx, cy, w, h])
                cls = 0 if ann['class'] == "orange" else 1
                dps_anno[filename].append((bb, cls))
        else:
            bb = np.array([])   # for bg
            cls = 2
            dps_anno[filename].append((bb, cls))

    return list(dps_anno.keys()), list(dps_anno.values())


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

    print("[INFO] Train labels Shape:", train_labels.shape)
    print("[INFO] Dev labels Shape:", dev_labels.shape)

    # Create the two iterators over the two datasets
    train_inputs = input_fn(True, train_filenames, train_labels, config_args)
    eval_inputs = input_fn(False, train_filenames, train_labels, config_args)


    # Define the model
    train_model_specs = model_fn('train', train_inputs, preset, config_args)
    eval_model_specs = model_fn('eval', eval_inputs, preset, config_args, reuse=True)

    print(train_model_specs['predictions'])
    print(train_model_specs['loss'])

    # Train the model
    train_and_evaluate(train_model_specs, eval_model_specs, 'model_dir', config_args)
