from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from a3c import *

import os
import sys
from config import config_init
import globalvar as GL
import numpy as np
from absl import flags

flags.DEFINE_string('f', '', 'kernel')
flags.FLAGS(sys.argv)

config_flags = config_init()
max_epoch = config_flags.max_episodes
map_name = config_flags.map
if config_flags.training:
    parallel = config_flags.parallel  # PARALLEL 指定开几个线程（几个游戏窗口在跑星际2）
    DEVICE = ['/gpu:' + dev for dev in config_flags.device.split(',')]
else:
    # PARALLEL = 1
    parallel = config_flags.parallel  # PARALLEL 指定开几个线程（几个游戏窗口在跑星际2）
    DEVICE = ['/cpu:0']

GL.global_init(parallel)
COUNTER = GL.get_episode_counter()
LOG = config_flags.log_path + config_flags.map + '/' + config_flags.net
SNAPSHOT = config_flags.snapshot_path + config_flags.map + '/' + config_flags.net
ANALYSIS = './DataForAnalysis/'
if not os.path.exists(LOG):
    os.makedirs(LOG)
if not os.path.exists(SNAPSHOT):
    os.makedirs(SNAPSHOT)
if not os.path.exists(ANALYSIS):
    os.makedirs(ANALYSIS)

run_a3c(max_epoch, map_name, parallel, config_flags, SNAPSHOT)