import argparse
import logging
import os
import json

import tensorflow as tf
import numpy as np

from model.model_fn import model_fn
from model.input_fn import input_fn
from model.training import train_and_evaluate
from model.utils import Params
from model.utils import set_logger
from model.ssdutils import get_preset_by_name


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help='Experiment directory containing params.json')
parser.add_argument('--data_dir', default='dataset/cone/train_dev',
                    help='Directory containing the dataset')
parser.add_argument('--restore_from', default=None,
                    help="optional, directory or file containing weights to reload before training")


if __name__ == "__main__":
    # set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load the parametes from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Check that we are not overwriting some previous experiment
    model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    overwritting = model_dir_has_best_weights and args.restore_from is None
    assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    model_dir = args.model_dir
    image_dir = os.path.join(data_dir, 'Images')
    label_dir = os.path.join(data_dir, 'Labels')

    # Create the two iterators over the two datasets
    train_inputs = input_fn(True, image_dir, label_dir, params)
    eval_inputs = input_fn(False, image_dir, label_dir,  params)

    # Define the model
    logging.info("Creating the model...")
    preset = get_preset_by_name('ssdmobilenet160')
    train_model_specs = model_fn('train', train_inputs, preset, params)
    eval_model_specs = model_fn('eval', eval_inputs, preset, params, reuse=True)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_specs, eval_model_specs, model_dir, params, args.restore_from)
