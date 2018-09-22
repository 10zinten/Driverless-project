import numpy as np
from math import sqrt, log, exp
from collections import namedtuple, defaultdict


Size    = namedtuple('Size',    ['w', 'h'])

SSDMap = namedtuple('SSDMap', ['size', 'scale', 'aspect_ratios'])
SSDPreset = namedtuple('SDDPreset', ['name', 'image_size', 'maps',
                                     'extra_scale', 'num_anchors'])

SSD_PRESETS = {
    'ssdmobilenet160': SSDPreset(name='mobilenet160',
                              image_size=Size(160, 160),
                              maps = [
                                  SSDMap(Size(10, 10), 0.1, [2, 1, 0.5, 1./3.]),
                                  SSDMap(Size( 5,  5), 0.267,  [2, 3, 0.5, 1./3.]),
                                  SSDMap(Size( 3,  3), 0.43, [2, 0.5]),
                                  SSDMap(Size( 1,  1), 0.5,   [2, 0.5]),
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
    # For each filter_map find the widths and heights of diff aspect ratio
    # Widths and heights of particular filter map are common for all the spacial location of this filter map
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
    # Using common widths and height of filermap, find x, y for different spacial location in filter map.
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

def abs2prop(xmin, xmax, ymin, ymax, img_size):
    """
    Convert the absolute min-max box bound to proportional center-width bounds.
    """
    w = float(xmax-xmin) + 1
    h = float(ymax-ymin) + 1
    cx = float(xmin) + w/2
    cy = float(ymin) + h/2
    w /= img_size.w
    h /= img_size.h
    cx /= img_size.w
    cy /= img_size.h

    return cx, cy, w, h


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
    arr[2] = log(bx[2] / an[2]) * 5
    arr[3] = log(bx[3] / an[3]) * 5

    return arr


def jaccard_overlap(bx, an):
    areaAn = (an[:, 1] - an[:, 0]+1) * (an[:, 3] - an[:, 2]+1)
    areaBx = (bx[1] - bx[0]+1) * (bx[3] - bx[2]+1)

    ixmin = np.maximum(bx[0], an[:, 0])
    ixmax = np.minimum(bx[1], an[:, 1])
    iymin = np.maximum(bx[2], an[:, 2])
    iymax = np.minimum(bx[3], an[:, 3])

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


def create_labels(preset, num_samples, num_classes, gts):
    """
    Create a dataset label out of ground truth.
    Shape: (num_anchors, num_classes+5)
    """

    def __process_overlap(dp_id, idx, score, gt, anchor, matches, num_classes):
        """
        i: datapoint id
        idx: index of overlapped anchor
        score: socre of overlapped acnhor
        """
        box, label = gt
        if idx in matches and matches[idx] >= score:
            return

        matches[idx] = score
        labels[dp_id, idx, 0:num_classes+1] = 0
        labels[dp_id, idx, label] = 1
        labels[dp_id, idx, num_classes+1:] = compute_location(box, anchor)


    # Initialized the necessary variable
    anchors = get_anchors_for_preset(preset)
    img_size = preset.image_size
    anchors_arr = anchors2array(anchors, img_size)
    labels = np.zeros((num_samples, anchors.shape[0], num_classes+5), dtype=np.float32)

    labels[:, :, num_classes] = 1    # bg
    labels[:, :, num_classes+1] = 0  # x offset
    labels[:, :, num_classes+2] = 0  # y offset
    labels[:, :, num_classes+3] = 0  # log width scale
    labels[:, :, num_classes+4] = 0  # log height scale

    # For every box compute the best match and all the matches above 0.5
    # Jaccard overlap
    overlaps = []
    for i in range(num_samples):
        # avoid the image with no objects, bbox is empty array
        if len(gts[i][0][0]) == 0:
            continue

        for box, _ in gts[i]:
            box_arr = box2array(box, img_size)
            overlaps.append(compute_overlap(box_arr, anchors_arr, 0.5))

        matches = {}
        for j, gt in enumerate(gts[i]):
            idxs, scores = overlaps[j]
            for idx, score in zip(idxs, scores):
                anchor = anchors[idx]
                __process_overlap(i, idx, score, gt, anchor, matches, num_classes)

    return labels

def decode_location(bx, an):
    bx[bx > 100] = 100 # only happends early training
    arr = np.zeros((4))

    arr[0] = bx[0]/10 * an[2] + an[0]
    arr[1] = bx[1]/10 * an[3] + an[1]
    arr[2] = exp(bx[2]/5) * an[2]
    arr[3] = exp(bx[3]/5) * an[3]

    return arr

def tf_decode_boxes(bx, an):
    pass

def decode_boxes(pred, anchors, conf_threshold=0.01, detections_cap=200):
    """
    Decode boxes from the result of ssd.
    """

    # Find the detections
    num_classes = pred.shape[1]-4
    bg_class = num_classes-1
    box_class = np.argmax(pred[:, :num_classes-1], axis=1)
    conf = pred[np.arange(len(pred)), box_class]

    if detections_cap is not None:
        detections = np.argsort(conf)[::-1][:detectionscap]
    else:
        detections = np.argsort(conf)[::-1]

    # Decode coordinates of each box with confidence over a threshold
    boxes = []
    for idx in detections:
        conf = pred[idx, box_class[idx]]
        if conf < conf_threshold:
            break

        box = decode_location(pred[idx, num_classes:], anchors[idx])
        cid = box_class[idx]
        boxes.append((conf, cid, box))

    return boxes


def non_max_suppression(boxes, overlap_th):
    # convert to absolute coordinates
    xmin, xmax, ymin, ymax = [], [], [], []
    img_size = Size(160, 160)

    for box in boxes:
        params = prop2abs(box[2], img_size)
        xmin.append(params[0])
        xmax.append(params[1])
        ymin.append(params[2])
        ymax.append(params[3])

    xmin = np.array(xmin)
    xmax = np.array(xmax)
    ymin = np.array(ymin)
    ymax = np.array(ymax)
    conf = np.array(boxes[0])

    # Compute the area of each box and sort the indices by conf level
    area = (xmax - xmin + 1) * (ymax - ymin + 1)
    idxs = np.argsort(conf)
    pick = []

    # Loop until we still have indices to precess:
    while len(idxs) > 0:
        # Grap the last index (ie. the most conf detection),
        # Remove it from the list of indeces to precess
        # Put on the list of picks
        last = idxs.shape[0]-1
        best_conf_idx = idxs[last]
        idxs = np.delete(idxs, last) # indices of remaining windows
        pick.append(best_conf_idx)
        suppress = []

        # Figure out the intersection with remaining windows
        ixmin = np.maximum(xmin[best_conf_idx], xmin[idxs])
        ixmax = np.minimum(xmax[best_conf_idx], xmax[idxs])
        iymin = np.maximum(ymin[best_conf_idx], ymin[idxs])
        iymax = np.minimum(ymax[best_conf_idx], ymax[idxs])

        w = np.maximum(0, ixmax - ixmin + 1)
        h = np.maximum(0, iymax - iymin + 1)
        intersection = w * h

        # Compute IOU and suppress indices with IOU higher a threshold
        union = area[i] + area[idxs] - intersection
        iou = intersection / unio
        overlap = iou > overlap_th
        suppress = np.nonzero(overlap)[0]
        idxs = np.delete(idxs, suppress)

    # Result the selected boxes
    selected = []
    for i in pick:
        selected.append(boxes[i])

    return selected


def suppress_overlaps(boxes):
    class_boxes = defaultdict(list)
    selected_boxes = []
    for box in boxes:
        class_boxes[box[1]].append(box)

    for cid, box in class_boxes.items():
        selected_boxes += non_max_suppression(v, 0.45)
    return selected_boxes
