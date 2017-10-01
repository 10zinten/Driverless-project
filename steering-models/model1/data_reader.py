'''
This module is used for loading training data into different train batch and
aslo for loading cv and test data.
'''

import numpy as np
from .preprocess import TRAIN_DIR, CV_DIR, TEST_DIR


class DataReader(object):
    def __init(self):
        self.train_set = np.load(TRAIN_DIR)

    def train_batch(self, size=64):
        first_index, last_index = 0, size
        while last_index <= len(self.train_set) + size:
            yield self.train_set[first_index: last_index, :, :, :]
            first_index, last_index = last_index, last_index + size
