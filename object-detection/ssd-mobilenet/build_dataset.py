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

def hw_bb(bb):
    return bb[0], bb[1], bb[0]+bb[2]-1, bb[1]+bb[3]-1
def process_ann(dp):
    for ann in dp['annotations']:
        xmin, ymin, xmax, ymax = hw_bb([ann['x'], ann['y'], ann['width'], ann['height']])
        ann['xmin'] = xmin
        ann['xmax'] = xmax
        ann['ymin'] = ymin
        ann['ymax'] = ymax

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

    process_ann(datapoint)

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

    output_dir_img = os.path.join(args.output_dir, 'Images')
    output_dir_labels = os.path.join(args.output_dir, 'Labels')

    if not os.path.exists(output_dir_img):
        os.mkdir(output_dir_img)
    else:
        print('[Warning]: output_dir {} already exists'.format(output_dir_img))

    if not os.path.exists(output_dir_labels):
        os.mkdir(output_dir_labels)
    else:
        print('[Warning]: output_dir {} already exists'.format(output_dir_labels))

    # Preprocess train, dev and test
    for split in ['train', 'dev', 'test']:
        print('Processing {} data'.format(split))
        for datapoint in tqdm(dataset[split]):
            resize_and_save(datapoint, output_dir_img, size=SIZE)

        with open(os.path.join(output_dir_labels, split + '.json'), 'w') as outfile:
            json.dump(dataset[split], outfile, indent=4)

        print('Done building dataset')
