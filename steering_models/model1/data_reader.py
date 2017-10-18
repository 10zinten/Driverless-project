'''
This module is used for loading training data into different train batch and
aslo for loading cv and test data.
'''

import numpy as np
from .preprocess import TRAIN_DIR, CV_DIR, TEST_DIR


def seperate_features_from_label(xs, ys, datasets):
    for data_point in datasets:
        xs.append(data_point[0])
        ys.append(data_point[1])


class DataReader(object):
    def __init(self):
        self.train_set = np.load(TRAIN_DIR)
        self.xs = []
        self.ys = []

        # seperating feature xs from label ys
        seperate_features_from_label(self.xs, self.ys, self.train_set)

    def train_batch(self, size=64):
        first_index, last_index = 0, size
        while last_index <= len(self.train_set) + size:
            yield np.array(self.xs[first_index: last_index]), np.array(self.ys[first_index: last_index])
            first_index, last_index = last_index, last_index + size

    def cv_set(self):
        cv_set = np.load(CV_DIR)
        # Loading angle from cv.csv file
        return cv_set   # [cv_set, csv]

    def test_set(self):
        test_set = np.load(TEST_DIR)
        xs = []
        ys = []
        seperate_features_from_label(xs, ys, test_set)
        return np.array(xs), np.array(ys)
