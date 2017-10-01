'''
Preprocess the images to zero mean and split them to Training, Cross Validation
and Test data and store them into the numpy array.
'''

import os
import numpy as np
from random import shuffle
from scipy.misc import imread, imresize


IMAGES_DIR = 'images/'
OUTPUT_DIR = 'data/'
TRAIN_DIR = 'data/train/train_set.npy'
CV_DIR = 'data/cross_validation/cv_set.py'
TEST_DIR = 'data/test/test_set.npy'

IMAGE_SIZE = (128, 128, 3)
TRAINING_SIZE = 0.98
CV_SIZE = 0.01
TEST_SIZE = 0.01


def resize():
    data = []
    for img in os.listdir(IMAGES_DIR):
        img = imresize(imread(IMAGES_DIR + img), IMAGE_SIZE)
        data.append(img)
    shuffle(data)
    return data


def split():
    data = resize()

    # training dataset
    first_index, last_index = 0, int(len(data) * TRAINING_SIZE)
    train = data[first_index: last_index]

    # Cross validation dataset
    first_index, last_index = last_index, last_index + int(len(data) * CV_SIZE)
    cv = data[first_index: last_index]

    # Test dataset
    first_index = last_index
    test = data[first_index: last_index]

    return train, cv, test


def preprocess():
    train, cv, test = split()
    np.save(TRAIN_DIR, train)
    np.save(CV_DIR, cv)
    np.save(TEST_DIR, test)
