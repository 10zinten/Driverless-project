from collections import namedtuple

import numpy as np
from math import sqrt

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

def get_anchor_for_preset(preset):
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

    print(box_sizes[-1])
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

if __name__ == "__main__":
    preset = get_preset_by_name('ssdmobilenet160')
    anchors = get_anchor_for_preset(preset)
    print(anchors)
    print(anchors.shape)
