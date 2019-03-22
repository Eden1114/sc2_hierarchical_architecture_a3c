from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pysc2.lib import actions
from pysc2.lib import features

# TODO: preprocessing functions for the following layers
_MINIMAP_PLAYER_ID = features.MINIMAP_FEATURES.player_id.index
_SCREEN_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_SCREEN_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_REQUIRED_MINIMAP = [0, 1, 3, 4, 5, 6]
# 0: height_map, 1: visibility_map, 2: creep, 3: camera, 4: player_id, 5: player_relative, 6: selected
_REQUIRED_SCREEN = [0, 1, 4, 5, 6, 7, 8, 9, 14, 15, 16]


# 0: height_map, 1: visibility_map, 2: creep, 3: power, 4: player_id, 5: player_relative, 6: unit_type,
# 7: selected, 8: unit_hit_points, 9: unit_hit_points_ratio, 10: unit_energy, 11: unit_energy_ratio,
# 12: unit_shields, 13: unit_shields_ratio, 14: unit_density, 15: unit_density_aa, 16: effects


def preprocess_minimap(minimap):
    minimap = np.array(minimap, dtype=np.float32)
    layers = []
    assert minimap.shape[0] == len(features.MINIMAP_FEATURES)
    for i in range(len(features.MINIMAP_FEATURES)):
        if i in _REQUIRED_MINIMAP:
            if i == _MINIMAP_PLAYER_ID:
                layers.append(minimap[i:i + 1] / features.MINIMAP_FEATURES[i].scale)
            elif features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:
                layers.append(minimap[i:i + 1] / features.MINIMAP_FEATURES[i].scale)
            else:
                layer = np.zeros([features.MINIMAP_FEATURES[i].scale, minimap.shape[1], minimap.shape[2]],
                                 dtype=np.float32)
                for j in range(features.MINIMAP_FEATURES[i].scale):
                    indy, indx = (minimap[i] == j).nonzero()
                    layer[j, indy, indx] = 1
                layers.append(layer)
    return np.expand_dims(np.concatenate(layers, axis=0), axis=0)


def preprocess_screen(screen):
    screen = np.array(screen, dtype=np.float32)
    layers = []
    assert screen.shape[0] == len(features.SCREEN_FEATURES)
    for i in range(len(features.SCREEN_FEATURES)):
        if i in _REQUIRED_SCREEN:
            if i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE:
                layers.append(screen[i:i + 1] / features.SCREEN_FEATURES[i].scale)
            elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
                layers.append(screen[i:i + 1] / features.SCREEN_FEATURES[i].scale)
            else:
                layer = np.zeros([features.SCREEN_FEATURES[i].scale, screen.shape[1], screen.shape[2]],
                                 dtype=np.float32)
                for j in range(features.SCREEN_FEATURES[i].scale):
                    indy, indx = (screen[i] == j).nonzero()
                    layer[j, indy, indx] = 1
                layers.append(layer)
    return np.expand_dims(np.concatenate(layers, axis=0), axis=0)


def minimap_channel():
    c = 0
    for i in range(len(features.MINIMAP_FEATURES)):
        if i in _REQUIRED_MINIMAP:
            if i == _MINIMAP_PLAYER_ID:
                c += 1
            elif features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:
                c += 1
            else:
                c += features.MINIMAP_FEATURES[i].scale
    return c


def screen_channel():
    c = 0
    for i in range(len(features.SCREEN_FEATURES)):
        if i in _REQUIRED_SCREEN:
            if i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE:
                c += 1
            elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
                c += 1
            else:
                c += features.SCREEN_FEATURES[i].scale
    return c
