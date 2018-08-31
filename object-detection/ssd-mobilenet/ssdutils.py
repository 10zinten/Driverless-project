from collections import namedtuple

import numpy as np

from utils import Size


SSDMap = namedtuple('SSDMap', ['size', 'scale', 'aspect_ratios'])
SSDPreset = namedtuple('SDDPreset', ['name', 'image_size', 'maps',
                                     'extra_scale', 'num_anchors'])

SSD_PRESETS = {
    'mobilenet160': SSDPreset(name='mobilenet160',
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


# default box paremeters both in terms proportional to image dimensions
Acchor = namedtuple('Anchor', ['center', 'size', 'x', 'y', 'scale', 'map'])

def get_preset_by_name(pname):
    if not pname in SSD_PRESETS:
        raise RuntimeError('No such preset:',pname)
    return SSD_PRESETS[pname]
