"""Evaluate the model"""

import argparse
import logging
import os

import tensorflow as tf

from model.input_fn import input_fn
from model.model_fn import model_fn
from model.evaluation import evaluate
from model.utils import parse_args, get_filenames_and_labels
from model.ssdutils import get_preset_by_name, create_labels



parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help='Experiment directory containing params.json')
parser.add_argument('--data_dir', default='datasets/cone',
                    help='Directory contraining the dataset')
parser.add_argument('--restore_from', default='best_weights',
                    help='Subdirectory of model dir or file containing the weights')

if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    # TODO: Load the parameters
    config_args = parse_args()

    # Create the inputs data pipeline
    data_dir = 'dataset/cone/'
    model_dir = 'experiments/basic_model'
    restore_from = 'best_weights'
    image_dir = os.path.join(data_dir, 'Images')
    label_dir = os.path.join(data_dir, 'Labels')

    test_filenames, test_labels = get_filenames_and_labels(image_dir, label_dir, 'test')

    # create ssd labels
    test_size = len(test_filenames)

    preset = get_preset_by_name('ssdmobilenet160')
    test_labels = create_labels(preset, test_size, 2, test_labels)

    print("[INFO] Test labels Shape:", test_labels.shape)

    # Create the two iterators over the two datasets
    test_inputs = input_fn(False, test_filenames, test_labels, config_args)

    # Define the model
    model_specs = model_fn('eval', test_inputs, preset, config_args)

    evaluate(model_specs, model_dir, config_args, restore_from)
