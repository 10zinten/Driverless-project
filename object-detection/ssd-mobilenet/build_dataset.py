import os
import json
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

SIZE = 160
PATH = ''

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help='Directory with label json file')
parser.add_argument('--output_dir', help='Where to write the new dataset')

def __to_square(array):
    pad = array[-1:, :, :]
    for _ in range((array.shape[1]-array.shape[0])):
        array = np.vstack((array, pad))
    return array

def resize_and_save(datapoint, output_dir, size=SIZE):
    """Rezize the image w/o maintaining aspect ratio
       and save image and lable json file to output_dir
    """
    filename = str(PATH/datapoint['filename'])
    img = Image.open(filename)
    img_array = np.array(img)
    img_array = __to_square(img_array)
    img = Image.fromarray(img_array)

    filename = filename.split('/')[-1]
    img.save(os.path.join(output_dir, filename))

    datapoint['filename'] = filename

if __name__ == '__main__':
    global PATH

    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # load the labels - constant filename
    PATH = Path(args.data_dir)
    trn_j = json.load((PATH/'labels.json').open())

    # split the dataset into 80% train, 10% dev and 10% test
    random.seed(230)
    random.shuffle(trn_j)

    split_1 = int(0.8 * len(trn_j))
    split_2 = int(0.9 * len(trn_j))
    train_set = trn_j[:split_1]
    dev_set = trn_j[split_1: split_2]
    test_set = trn_j[split_2:]

    dataset = {'train': train_set,
               'dev': dev_set,
               'test': test_set}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print('[Warning]: output_dir {} already exists'.format(args.output_dir))

    # Preprocess train, dev and test
    for split in ['train', 'dev', 'test']:
        output_dir_split = os.path.join(args.output_dir, '{}_dir'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print('[Warnin]: dir {} already exists'.format(output_dir_split))

        print('Processing {} data, saving preprocessed data to {}'.format(split, output_dir_split))
        for datapoint in tqdm(dataset[split]):
            resize_and_save(datapoint, output_dir_split, size=SIZE)

        with open(os.path.join(output_dir_split, 'labels.json'), 'w') as outfile:
            json.dump(dataset[split], outfile, indent=4)

        print('Done building dataset')
