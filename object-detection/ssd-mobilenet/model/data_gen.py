'Tran and validation data generator'

import os
import random
from copy import copy

import cv2
import numpy as np

from model.utils import get_filenames_and_labels
from model.ssdutils import get_anchors_for_preset, get_preset_by_name, anchors2array
from model.ssdutils import box2array, compute_overlap, compute_location


preset = get_preset_by_name('ssdmobilenet160')

class Transform:
    def __init__(self, **kwargs):
        for arg, val in kwargs.items():
            setattr(self, arg, val)
        self.initialized = False

class ImageLoaderTransform(Transform):
    """
    load and image from the filename
    """
    def __call__(self, filename, label, gt):
        return cv2.imread(filename), label, gt

    def __repr__(self):
        return "ImageLoader Transform"

###############################################################################
#                          Photometric Distortions                            #
###############################################################################
class BrightnessTransform(Transform):
    """
    Transforms the image brightness
    Parameters: delta
    """
    def __call__(self, data, label, gt):
        data = data.astype(np.float32)
        delta = random.randint(-self.delta, self.delta)
        data += delta
        data[data > 255] = 255
        data[data < 0] = 0
        data = data.astype(np.uint8)
        return data, label, gt

    def __repr__(self):
        return "Brightness Transform"

class ConstrastTransform(Transform):
    """
    Transform image constrast
    Parameters: lower, upper
    """
    def __call__(self, data, label, gt):
        data = data.astype(np.float32)
        delta = random.uniform(self.lower, self.upper)
        data *= delta
        data[data > 255] = 255
        data[data < 0] = 0
        data = data.astype(np.uint8)
        return data, label, gt

    def __repr__(self):
        return "Constrast Transform"

class HueTransform(Transform):
    """
    Transform hue
    Parameters: delta
    """
    def __call__(self, data, label, gt):
        data = cv2.cvtColor(data, cv2.COLOR_BGR2HSV)
        data = data.astype(np.float32)
        delta = random.randint(-self.delta, self.delta)
        data[0] += delta
        data[0][data[0] > 180] -= 180
        data[0][data[0] < 0] += 180
        data = data.astype(np.uint8)
        data = cv2.cvtColor(data, cv2.COLOR_HSV2BGR)
        return data, label, gt

    def __repr__(self):
        return "Hue Transform"

class SaturationTransform(Transform):
    """
    Transform saturation
    Parameters: lower, upper
    """
    def __call__(self, data, label, gt):
        data = cv2.cvtColor(data, cv2.COLOR_BGR2HSV)
        data = data.astype(np.float32)
        delta = random.uniform(self.lower, self.upper)
        data[1] *= delta
        data[1][data[1]>180] -= 180
        data[1][data[1]<0] +=180
        data = data.astype(np.uint8)
        data = cv2.cvtColor(data, cv2.COLOR_HSV2BGR)
        return data, label, gt

    def __repr__(self):
        return "Saturation Transform"

class ChannelsReorderTransform(Transform):
    def __call__(self, data, label, gt):
        channels = [0, 1, 2]
        random.shuffle(channels)
        return data[:, :, channels], label, gt

    def __repr__(self):
        return "Channels Reorder Transform"

###############################################################################
#                          Geometric Distortions                              #
###############################################################################
class HorizontalFlip(Transform):
    """
    Tranfrom the image and bboxes with horizontal flip
    """
    def __call__(self, image, label, gt):
        img_center = np.array(image.shape[: 2])[::-1] / 2
        img_center = np.hstack((img_center, img_center)).astype(np.uint8)
        image = image[:, ::-1, :]
        gt[:, [0, 1]] += 2*(img_center[[0, 1]] - gt[:, [0, 1]])

        return image, label, gt

    def __repr__(self):
        return "Horizontal Filp"


class RandomTransform(Transform):
    """
    Call another transfrom with a given probability
    Parameters: prob, transform
    """
    def __call__(self, data, label, gt):
        p  = random.uniform(0, 1)
        if p < self.prob:
            return self.transform(data, label, gt)
        return data, label, gt

    def __repr__(self):
        return repr(self.transform)

class ComposeTransform(Transform):
    """
    call a bunch of transforms serially
    Parametes: transforms
    """
    def __call__(self, data, label, gt):
        args = (data, label, gt)
        for transform in self.transforms:
            args = transform(*args)
        return args

class TransformPickerTransform(Transform):
    """
    Call a randomly chosen transform from the list
    Parameters: transforms
    """
    def __call__(self, data, label, gt):
        self.pick = random.randint(0, len(self.transforms)-1)
        return self.transforms[self.pick](data, label, gt)

    def __repr__(self):
        return repr(self.transforms[self.pick])


###############################################################################
#                     Label Creator Transform                                 #
###############################################################################

def process_overlap(idx, score, gt, anchor, vec, matches, num_classes):
    """
    i: datapoint id
    idx: index of overlapped anchor
    score: socre of overlapped acnhor
    """
    box, label = gt[:-1], gt[-1]
    if idx in matches and matches[idx] >= score:
        return

    matches[idx] = score
    vec[idx, 0:num_classes+1] = 0
    vec[idx, label] = 1
    vec[idx, num_classes+1:] = compute_location(box, anchor)

class LabelCreatorTransform(Transform):
    """
    Create a label vector out of a ground truth of a sample.
    Parameters: preset, num_classes
    """
    def initialize(self):
        self.anchors = get_anchors_for_preset(self.preset)
        self.img_size = self.preset.image_size
        self.anchors_arr = anchors2array(self.anchors, self.img_size)
        self.initialized = True

    def __call__(self, data, label, gt):
        if not self.initialized:
            self.initialize()

        vec = np.zeros((len(self.anchors), self.num_classes+5), dtype=np.float32)
        vec[:, self.num_classes] = 1    # bg
        vec[:, self.num_classes+1] = 0  # x offset
        vec[:, self.num_classes+2] = 0  # y offset
        vec[:, self.num_classes+3] = 0  # log width scale
        vec[:, self.num_classes+4] = 0  # log height scale

        overlaps = []
        for box in gt:
            box_arr = box2array(box[:-1], self.img_size)
            overlaps.append(compute_overlap(box_arr, self.anchors_arr, 0.1))
        print(overlaps)

        matches = {}
        for j, gt in enumerate(gt):
            idxs, scores = overlaps[j]
            for idx, score in zip(idxs, scores):
                anchor = self.anchors[idx]
                process_overlap(idx, score, gt, anchor, vec, matches, self.num_classes)

        return data, vec, gt

    def __repr__(self):
        return "Label Creator Transform"



def build_train_transforms(preset, num_classes):

    ####  Photomatic Distortions  ###
    brightness = BrightnessTransform(delta=100)
    random_brightness = RandomTransform(prob=0.5, transform=brightness)

    constrast = ConstrastTransform(lower=0.5, upper=1.8)
    random_constrast = RandomTransform(prob=0.5, transform=constrast)

    hue = HueTransform(delta=100)
    random_hue = RandomTransform(prob=0.5, transform=hue)

    saturation = SaturationTransform(lower=0.5, upper=1.8)
    random_saturation = RandomTransform(prob=0.5, transform=saturation)

    channels_reorder = ChannelsReorderTransform()
    random_channels_reorder = RandomTransform(prob=0.3, transform=channels_reorder)

    # Compositions of image distortions
    distort_list = [
        random_constrast,
        random_hue,
        random_saturation,
    ]

    distort_1 = ComposeTransform(transforms=distort_list[:-1])
    distort_2 = ComposeTransform(transforms=distort_list[1:])
    distort_comp = [distort_1, distort_2]
    distort = TransformPickerTransform(transforms=distort_comp)

    ### Geometric Distortions ###
    horizontal_flip = HorizontalFlip()
    random_horizontal_flip = RandomTransform(prob=0.5, transform=horizontal_flip)

    transforms = [
            ImageLoaderTransform(),
            random_brightness,
            distort,
            random_channels_reorder,
            random_horizontal_flip,
            LabelCreatorTransform(preset=preset, num_classes=num_classes)
        ]

    return transforms

def build_val_transforms(preset, num_classes):

    transforms = [
            ImageLoaderTransform(),
            LabelCreatorTransform(preset=preset, num_classes=num_classes)
        ]

    return transforms

class TrainingData:

    def __init__(self, image_dir, label_dir, param):
        train_filenames, train_label = get_filenames_and_labels(image_dir, label_dir, 'train')
        nones = [None] * len(train_filenames)
        train_samples = list(zip(train_filenames, nones, train_label))
        val_filenames, val_label = get_filenames_and_labels(image_dir, label_dir, 'dev')
        nones = [None] * len(val_filenames)
        val_samples = list(zip(val_filenames, nones, val_label))


        self.preset = preset = get_preset_by_name('ssdmobilenet160')
        self.num_classes = 2
        self.train_tfs = build_train_transforms(self.preset, self.num_classes)
        self.val_tfs = build_val_transforms(self.preset, self.num_classes)
        self.train_generator  = self.__build_generator(train_samples, self.train_tfs)
        self.val_generator    = self.__build_generator(val_samples, self.val_tfs)
        self.num_train = len(train_samples)
        self.num_val = len(val_samples)
        self.train_samples = list(zip(train_filenames, train_label))
        self.val_samples = list(zip(val_filenames, val_label))

    def __build_generator(self, all_samples_, transforms):

        def run_transforms(sample):
            args = sample
            for transform in transforms:
                args = transform(*args)

            return args

        def process_samples(samples):
            images, labels, gt_boxes = [], [], []
            for sample in samples:
                image, label, gt = run_transforms(sample)
                images.append(image.astype(np.float32))
                labels.append(label)
                gt_boxes.append(gt)

            images = np.array(images, dtype=np.float32)

            return images, labels, gt_boxes


        def gen_batch(batch_size):
            all_samples = copy(all_samples_)
            random.shuffle(all_samples)

            for offset in range(0, len(all_samples), batch_size):
                samples = all_samples[offset: offset + batch_size]
                images, labels, gt_boxes = process_samples(samples)

                for transform in transforms:
                    print("[INFO] {} applied ... ok".format(transform))
                print()

                yield images, labels, gt_boxes

        return gen_batch
