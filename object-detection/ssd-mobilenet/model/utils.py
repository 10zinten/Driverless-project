import os
import sys
import json
import argparse
import logging

import numpy as np
import cv2
import pickle
from collections import namedtuple, defaultdict
from easydict import EasyDict as edict

from model.ssdutils import abs2prop, prop2abs


Size    = namedtuple('Size',    ['w', 'h'])

class Params():
    """class that loads hyperparameters from a json file."""

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """save parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, ff, indent=4)

    def update(self, json_path):
        """load parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # convert the values to float for json (it doesn't accept np.array, np.float
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


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

        filenames = list(dps_anno.keys())
        labels = list(map(lambda k: dps_anno[k], filenames))

    return filenames, labels


def draw_box(img, box):
    img_size = Size(img.shape[1], img.shape[0])
    xmin, xmax, ymin, ymax = prop2abs(box[2], img_size)
    img_box = np.copy(img)
    cv2.rectangle(img_box, (xmin, ymin), (xmax, ymax), (1, 1, 1), 2)
    # cv2.rectangle(img_box, (xmin-1, ymin), (xmax+1, ymin-20), color, cv2.FILLED)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(img_box, box.label, (xmin+5, ymin-5), font, 0.5,
    #             (255, 255, 255), 1, cv2.LINE_AA)
    alpha = 0.8
    cv2.addWeighted(img_box, alpha, img, 1.-alpha, 0, img)
