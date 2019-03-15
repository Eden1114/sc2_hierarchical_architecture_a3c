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
max_episode = config_flags.max_episodes
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

run_a3c(max_episode, map_name, parallel, config_flags, SNAPSHOT)

# 最终数据记录
for i in range(parallel + 1):
    np.save("./DataForAnalysis/reward_list_thread_" + str(i) + ".npy", GL.get_value(i, "reward_list"))
    np.save("./DataForAnalysis/victory_or_defeat_thread_" + str(i) + ".npy", GL.get_value(i, "victory_or_defeat"))
    np.save("./DataForAnalysis/victory_or_defeat_self_thread_" + str(i) + ".npy",
            GL.get_value(i, "victory_or_defeat_self"))
    np.save("./DataForAnalysis/episode_score_list_thread_" + str(i) + ".npy", GL.get_value(i, "episode_score_list"))
print('Fin. ')
