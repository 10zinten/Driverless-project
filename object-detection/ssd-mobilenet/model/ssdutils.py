from collections import namedtuple

import numpy as np
from math import sqrt, log

from utils import Size


SSDMap = namedtuple('SSDMap', ['size', 'scale', 'aspect_ratios'])
SSDPreset = namedtuple('SDDPreset', ['name', 'image_size', 'maps',
                                     'extra_scale', 'num_anchors'])

SSD_PRESETS = {
    'ssdmobilenet160': SSDPreset(name='mobilenet160',
                              image_size=Size(160, 160),
                              maps = [
                                  SSDMap(Size(10, 10), 0.375, [2, 3, 0.5, 1./3.]),
                                  SSDMap(Size( 5,  5), 0.55,  [2, 3, 0.5, 1./3.]),
                                  SSDMap(Size( 3,  3), 0.725, [2, 0.5]),
                                  SSDMap(Size( 1,  1), 0.9,   [2, 0.5]),
                              ],
                              extra_scale = 107.5,
                              num_anchors = 790)
}


def get_preset_by_name(pname):
    if not pname in SSD_PRESETS:
        raise RuntimeError('No such preset:',pname)
    return SSD_PRESETS[pname]

def get_anchors_for_preset(preset):
    """
    Create the default (anchor) boxes for the given SSD preset
    Anchor format: (x, y, w, h)
    """

    # Compute the with and heights of the anchor boxes for every scale
    box_sizes = []
    for i in range(len(preset.maps)):
        map_param = preset.maps[i]
        scale = map_param.scale
        aspect_ratios = [1] + map_param.aspect_ratios
        aspect_ratios = list(map(lambda x: sqrt(x), aspect_ratios))

        sizes = []
        for ratio in aspect_ratios:
            w = scale * ratio
            h = scale / ratio
            sizes.append((w, h))

        if i < len(preset.maps)-1:
            s_prime = sqrt(scale*preset.maps[i+1].scale)
        else:
            s_prime = sqrt(scale*preset.extra_scale)
        sizes.append((s_prime, s_prime))
        box_sizes.append(sizes)

    # compute the actual boxes for every scale and feature map
    anchors = []
    for k in range(len(preset.maps)):
        fk = preset.maps[k].size[0]
        for size in box_sizes[k]:
            for j in range(fk):
                y = (j+0.5) / float(fk)
                for i in range(fk):
                    x = (i+0.5) / float(fk)
                    anchors.append([x, y, size[0], size[1]])

    return np.array(anchors)

def prop2abs(an, img_size):
    """
    Convert prop center-width bounds to absolute min-max bounds.
    """
    try:
        x, y, w, h = an[0], an[1], an[2], an[3] # anchors
    except:
        return
    aw = w*img_size.w
    ah = h*img_size.h
    cx = x*img_size.w
    cy = y*img_size.h

    return int(cx-aw/2), int(cx+aw/2), int(cy-ah/2), int(cy+ah/2)


def anchors2array(anchors, img_size):
    """
    Compute a absolute anchor params (xmin, xmax, ymin, ymax) from proportional
    params(x, y, w, h) of given anchor.
    """
    arr = np.zeros((anchors.shape))
    for i, anchor in enumerate(anchors):
        xmin, xmax, ymin, ymax = prop2abs(anchor, img_size)
        arr[i] = np.array([xmin, xmax, ymin, ymax])

    return arr


def box2array(box, img_size):
    xmin, xmax, ymin, ymax = prop2abs(box, img_size)
    return np.array([xmin, xmax, ymin, ymax])


def compute_location(bx, an):
    arr = np.zeros((4))
    arr[0] = (bx[0] - an[0]) / an[2] * 10
    arr[1] = (bx[1] - an[1]) / an[3] * 10
    arr[2] = log(box[2] / an[2]) * 5
    arr[3] = log(box[3] / an[3]) * 5

    return arr


def jaccard_overlap(bx, an):
    areaAn = (an[:, 1] - an[:, 0]+1) * (an[:, 3] - an[:, 2]+1)
    areaBx = (bx[1] - bx[0]+1) * (bx[3] - bx[4]+1)

    ixmin = np.maximum(bx[0], an[0])
    ixmax = np.minimum(bx[1], an[1])
    iymin = np.maximum(bx[2], an[2])
    iymax = np.manimum(bx[3], an[3])

    w = np.maximum(0, ixmax-ixmin+1)
    h = np.maximum(0, iymax-iymin+1)

    iarea = w*h
    union = areaAn + areaBx - iarea

    return iarea / union


def compute_overlap(box_arr, anchors_arr, threshold):
    iou = jaccard_overlap(box_arr, anchors_arr)
    overlap = iou > threshold

    overlap_idx = np.nonzero(overlap)[0]

    return overlap_idx, iou[overlap_idx]


def create_labels(preset, num_batch, num_classes, gt_boxes):
    """
    Create a dataset label out of ground truth.
    Shape: (num_anchors, num_classes+5)
    """

    def __process_overlap(dp_id, idx, score, box, anchor, matches, num_classes, labels):
        """
        i: datapoint id
        idx: index of overlapped anchor
        score: socre of overlapped acnhor
        """

        if idx in matches and matches[idx] >= score:
            return

        matches[idx] = score
        labels[dp_id, idx, 0:num_classes+1] = 0
        labels[dp_id, idx, box[1]] = 1
        labels[dp_id, idx, num_classes+1:] = compute_location(box, anchor)


    # Initialized the necessary variable
    anchors = get_anchors_for_preset(preset)
    print(anchors.shape)
    print(len(gt_boxes))
    img_size = preset.image_size
    anchors_arr = anchors2array(anchors, img_size)
    labels = np.zeros((num_batch, anchors.shape[0], num_classes+5), dtype=np.float32)

    labels[:, :, num_classes] = 1    # bg
    labels[:, :, num_classes+1] = 0  # x offset
    labels[:, :, num_classes+2] = 0  # y offset
    labels[:, :, num_classes+3] = 0  # log width scale
    labels[:, :, num_classes+4] = 0  # log height scale

    # For every box compute the best match and all the matches above 0.5
    # Jaccard overlap
    overlaps = []
    for i in range(num_batch):
        for boxes in gt_boxes[i]:
            box_arr = box2array(box, img_size)
            overlaps.append(compute_overlap(box_arr, anchor_arr, 0.5))

        matches = {}
        for j, box in enumerate(gt_boxes[i]):
            for idx, score in overlaps[j]:
                anchor = anchors[idx]
                process_overlap(i, idx, score, box, anchor, matches, num_classes, labels)

    return labels

if __name__ == "__main__":
    preset = get_preset_by_name('ssdmobilenet160')

    # Create sysnthetic gt_box
    boxes = np.random.rand(3, 4)
    labels = [0, 1, 0]
    gt_boxes = list(zip(boxes, labels))

    labels = create_labels(preset, 1, 2, gt_boxes)
    print(labels.shape)

