import os
import sys
import json
import argparse

import pickle
import numpy as np
import tensorflow as tf
from pprint import pprint
from collections import namedtuple, defaultdict
from easydict import EasyDict as edict


from model.ssdutils import abs2prop


Size    = namedtuple('Size',    ['w', 'h'])


def parse_args():
    """
    Parse the arguments of the program
    :return: (config_args)
    :rtype: tuple
    """
    # Create a parser
    parser = argparse.ArgumentParser(description="MobileNet TensorFlow Implementation")
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
    parser.add_argument('--config', default=None, type=str, help='Configuration file')

    # Parse the arguments
    args = parser.parse_args()

    # Parse the configurations from the config json file provided
    try:
        if args.config is not None:
            with open(args.config, 'r') as config_file:
                config_args_dict = json.load(config_file)
        else:
            print("Add a config file using \'--config file_name.json\'", file=sys.stderr)
            exit(1)

    except FileNotFoundError:
        print("ERROR: Config file not found: {}".format(args.config), file=sys.stderr)
        exit(1)
    except json.decoder.JSONDecodeError:
        print("ERROR: Config file is not a proper JSON file!", file=sys.stderr)
        exit(1)

    config_args = edict(config_args_dict)

    return config_args


def create_experiment_dirs(exp_dir):
    """
    Create Directories of a regular tensorflow experiment directory
    :param exp_dir:
    :return summary_dir, checkpoint_dir:
    """
    experiment_dir = os.path.realpath(os.path.join(os.path.dirname(__file__))) + "/experiments/" + exp_dir + "/"
    summary_dir = experiment_dir + 'summaries/'
    checkpoint_dir = experiment_dir + 'checkpoints/'
    dirs = [summary_dir, checkpoint_dir]
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        print("Experiment directories created!")
        # return experiment_dir, summary_dir, checkpoint_dir, output_dir, test_dir
        return experiment_dir, summary_dir, checkpoint_dir
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def calculate_flops():
    # Print to stdout an analysis of the number of floating point operations in the
    # model broken down by individual operations.
    tf.profiler.profile(
        tf.get_default_graph(),
        options=tf.profiler.ProfileOptionBuilder.float_operation(), cmd='scope')

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

