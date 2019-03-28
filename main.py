from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import importlib
import threading
from absl import app
from absl import flags
from pysc2 import maps
from pysc2.lib import stopwatch
import tensorflow as tf

from run_thread import run_thread
from config import config_init
import globalvar as GL
import numpy as np

LOCK = threading.Lock()
# DHN add:
UPDATE_ITER_LOW = 10  # 经历多少个step以后更新下层网络，10差不多是游戏里的4s少一点
UPDATE_ITER_HIGH = UPDATE_ITER_LOW * 2  # 经历多少个step以后更新上层网络，20差不多是游戏里的18s少一点

FLAGS = config_init()
FLAGS(sys.argv)
if FLAGS.training:
    PARALLEL = FLAGS.parallel  # PARALLEL 指定开几个线程（几个游戏窗口在跑星际2）
    DEVICE = ['/gpu:' + dev for dev in FLAGS.device.split(',')]
else:
    # PARALLEL = 1
    PARALLEL = FLAGS.parallel  # PARALLEL 指定开几个线程（几个游戏窗口在跑星际2）
    DEVICE = ['/cpu:0']

GL.global_init(PARALLEL)
COUNTER = GL.get_episode_counter()
LOG = FLAGS.log_path + FLAGS.map + '/' + FLAGS.net
SNAPSHOT = FLAGS.snapshot_path + FLAGS.map + '/' + FLAGS.net
ANALYSIS = './DataForAnalysis/'
if not os.path.exists(LOG):
    os.makedirs(LOG)
if not os.path.exists(SNAPSHOT):
    os.makedirs(SNAPSHOT)
if not os.path.exists(ANALYSIS):
    os.makedirs(ANALYSIS)


# main.py负责读取config参数，创建与监控thread，日志和数据记录
def _main(unused_argv):
    """Run agents"""
    stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace  # 应该是开启类似计时时钟这样的观测量
    stopwatch.sw.trace = FLAGS.trace
    maps.get(FLAGS.map)  # Assert the map exists.
    # Setup agents
    agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)  # 经过两行操作后，agent_cls就相当于A3CAgent类了，可用其构造对象
    agents = []
    for i in range(PARALLEL):
        # 用agent_cls(A3CAgent)构造对象（调用了a3c_agent文件__init__构造函数）
        agent = agent_cls(FLAGS.training, FLAGS.minimap_resolution, FLAGS.screen_resolution)
        # 现在agent就是一个被生成好的A3CAgent对象了，利用build_model创建所有需要的tf节点
        agent.build_model(i > 0, DEVICE[i % len(DEVICE)])
        agents.append(agent)  # agents是多个A3CAgent对象合集（如果PARALLEL大于1的话，不然就只有一个对象在里面）
    config = tf.ConfigProto(allow_soft_placement=True)  # 允许tf自动选择一个存在并且可用的设备来运行操作
    config.gpu_options.allow_growth = True  # 动态申请显存
    sess = tf.Session(config=config)
    summary_writer = tf.summary.FileWriter(LOG)  # 记录日志，供tensorboard使用
    for i in range(PARALLEL):
        agents[i].setup(sess, summary_writer)  # setup各个agent，即 将唯一的sess和summary_writer赋予每个agent的进程
    # agent就是“包含agents.append(agent)的循环”当中的那个局部变量agent，局部变量能够在外部使用，大概是python(3)神奇的特性...所以agent就是最后一个创建的A3CAgent对象
    agent.initialize()  # run(tf.global_variables_initializer())以初始化每个agent中的tf图
    # 模型读取
    if not FLAGS.training:  # 若不是训练模式
        _ = agent.load_model(SNAPSHOT)
        COUNTER = 0
        GL.set_episode_counter(COUNTER)
        print("Non-training Mode")
    if FLAGS.continuation:  # 若是继续训练，则利用原有数据（训练好的参数，存在了snapshot文件夹里）进行训练
        COUNTER = agent.load_model(SNAPSHOT)
        GL.set_episode_counter(COUNTER)
        # 全局变量COUNTER记录的是当前所有线程加在一起，总共完成的回合数
    # Run threads
    threads = []
    for i in range(PARALLEL - 1):  # 建立PARALLEL - 1个线程并运行
        t = threading.Thread(target=run_thread, args=(
            agents[i], FLAGS.map, False, i, FLAGS, LOCK))  # threading是python自己的线程模块，参数1为线程运行的函数名称，参数2为该函数需要的参数
        threads.append(t)
        t.daemon = True
        t.start()
        time.sleep(5)
    run_thread(agents[-1], FLAGS.map, FLAGS.render, PARALLEL - 1, FLAGS, LOCK)
    # 序号-1代表最后一个agent 这个线程运行时将可视化feature map（因为最后一个参数FLAGS.render为True，之前的几个线程改参数为False）
    for t in threads:
        t.join()  # 必须写 才能正常运行多线程
    if FLAGS.profile:
        print(stopwatch.sw)
    # 数据记录
    for i in range(PARALLEL + 1):
        np.save("./DataForAnalysis/reward_list_thread_" + str(i) + ".npy", GL.get_value(i, "reward_list"))
        np.save("./DataForAnalysis/victory_or_defeat_thread_" + str(i) + ".npy", GL.get_value(i, "victory_or_defeat"))
        np.save("./DataForAnalysis/victory_or_defeat_self_thread_" + str(i) + ".npy",
                GL.get_value(i, "victory_or_defeat_self"))
        np.save("./DataForAnalysis/episode_score_list_thread_" + str(i) + ".npy", GL.get_value(i, "episode_score_list"))
    print('Fin. ')


if __name__ == "__main__":
    app.run(_main)
