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
from pysc2.env import available_actions_printer
from pysc2.env import sc2_env
from pysc2.lib import stopwatch
import tensorflow as tf

from run_loop import run_loop
import agents.globalvar as GL

COUNTER = 0
LOCK = threading.Lock()

#DHN add:
UPDATE_ITER_LOW = 10                    # 经历多少个step以后更新下层网络，10差不多是游戏里的4s少一点
UPDATE_ITER_HIGH = UPDATE_ITER_LOW * 2   # 经历多少个step以后更新上层网络，20差不多是游戏里的18s少一点

FLAGS = flags.FLAGS           # 定义超参数
flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_bool("continuation", False, "Continuously training.")
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate for training.")
flags.DEFINE_float("discount", 0.99, "Discount rate for future rewards.")
flags.DEFINE_integer("max_steps", int(1e5), "Total steps for training.")    # 这里的step指的是训练的最大回合数，而不是回合episode里的那个step
flags.DEFINE_integer("snapshot_step", int(1e3), "Step for snapshot.")
flags.DEFINE_string("snapshot_path", "./snapshot/", "Path for snapshot.")
flags.DEFINE_string("log_path", "./log/", "Path for log.")
flags.DEFINE_string("device", "0", "Device for training.")

#flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")    # 该工程原代码
flags.DEFINE_string("map", "Simple64", "Name of a map to use.")         # 2018/08/03: Simple64枪兵互拼新加代码

flags.DEFINE_bool("render", True, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 64, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64, "Resolution for minimap feature layers.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")   # APM参数，step_mul为8相当于APM180左右

flags.DEFINE_string("agent", "agents.a3c_agent.A3CAgent", "Which agent to run.")
flags.DEFINE_string("net", "fcn", "atari or fcn.")
flags.DEFINE_string("agent_race", 'terran', "Agent's race.")

# 2018/08/03: Simple64枪兵互拼新加代码
flags.DEFINE_string("agent2", "Bot", "Second agent, either Bot or agent class.")
flags.DEFINE_enum("agent2_race", "terran", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 2's race.")
flags.DEFINE_enum("difficulty", "very_easy", sc2_env.Difficulty._member_names_,  # pylint: disable=protected-access
                  "If agent2 is a built-in Bot, it's strength.")

flags.DEFINE_integer("max_agent_steps", 10000, "Total agent steps.")       # 这里的step指的是回合episode里的那个step
flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")
flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")

FLAGS(sys.argv)
if FLAGS.training:  # 疑问：为啥训练模式跑gpu，回合里最大步数为60步；非训练模式跑cpu，回合里最大步数为100000步？
  PARALLEL = FLAGS.parallel   # PARALLEL 指定开几个线程（几个游戏窗口在跑星际2）
  MAX_AGENT_STEPS = FLAGS.max_agent_steps       # 回合里agent的最大步数
  DEVICE = ['/gpu:'+dev for dev in FLAGS.device.split(',')]
else:
  PARALLEL = 1
  MAX_AGENT_STEPS = 1e5         # 回合里agent的最大步数
  DEVICE = ['/cpu:0']

GL.global_init(PARALLEL)

LOG = FLAGS.log_path+FLAGS.map+'/'+FLAGS.net
SNAPSHOT = FLAGS.snapshot_path+FLAGS.map+'/'+FLAGS.net
if not os.path.exists(LOG):
  os.makedirs(LOG)
if not os.path.exists(SNAPSHOT):
  os.makedirs(SNAPSHOT)


def run_thread(agent, map_name, visualize, ind_thread):  # A3CAgent对象，地图名（字符串），是否可视化feature map（布尔值）
  players = [sc2_env.Agent(sc2_env.Race[FLAGS.agent_race])]
  players.append(sc2_env.Bot(sc2_env.Race[FLAGS.agent2_race], sc2_env.Difficulty[FLAGS.difficulty]))  # 2018/08/03: 2018/08/03: Simple64枪兵互拼新加代码
  agent_interface = sc2_env.parse_agent_interface_format(       # 增加一个player的代码应该在run_thread开头这里修改
    feature_screen=FLAGS.screen_resolution,
    feature_minimap=FLAGS.minimap_resolution)
  with sc2_env.SC2Env(
          map_name=map_name,
          players=players,
          step_mul=8,
          agent_interface_format=agent_interface,
          visualize=visualize) as env:
    env = available_actions_printer.AvailableActionsPrinter(env)

    # Only for a single player!
    counter = 0
    replay_buffer_1 = []  # 1供下层网络更新时使用
    replay_buffer_2 = []  # 2供上层网络更新时使用
    dir_high_buffer = []

    # 下行中的run_loop是个生成器，for循环每次进入到run_loop里，得到yield后返回，继续进行循环体里的语句，for循环再次进入run_loop后从run_loop的yield的下一条语句开始执行，执行到yield再次返回，继续执行循环体语句...
    for recorder, is_done, stepsInOneEp, call_step_low,macro_type,coord_type in run_loop([agent], env, MAX_AGENT_STEPS, ind_thread):   # 将agent对象存入[]再作为参数传递进run_loop生成器里，recorder是一个三元列表

      if FLAGS.training:    # 这里是if FLAGS.training，但后面并没有if not FLAGS.training。即若是非训练模式（restore了以前的网络参数），则不再进行网络参数的更新
        if call_step_low == 1:
          replay_buffer_1.append(recorder)
        replay_buffer_2.append(recorder)
        dir_high_buffer.append([GL.get_value(ind_thread, "dir_high")])
        if is_done:     # 若为训练模式
          with LOCK:    # 使用线程锁（跟java类似，应用于不同线程会调用相同资源的情况），给Counter和counter加一
            global COUNTER
            COUNTER += 1
            counter = COUNTER
          # Learning rate schedule
          # learning_rate = FLAGS.learning_rate * (1 - 0.9 * counter / FLAGS.max_steps)   # 根据当前进行完的回合数量修改学习速率（减小）
          # agent.update(replay_buffer, FLAGS.discount, learning_rate, counter)

        # 更新下层网络
        # if stepsInOneEp % UPDATE_ITER_LOW == 0 or is_done:
        if call_step_low == 1:
          learning_rate_a_low = FLAGS.learning_rate * (1 - 0.9 * counter / FLAGS.max_steps)   # 根据当前进行完的回合数量修改学习速率（减小）
          learning_rate_c_low = FLAGS.learning_rate * (1 - 0.9 * counter / FLAGS.max_steps)   # 根据当前进行完的回合数量修改学习速率（减小）
          agent.update_low(ind_thread,replay_buffer_1, FLAGS.discount, learning_rate_a_low, learning_rate_c_low, counter,macro_type,coord_type)

          # time.sleep(2)
          replay_buffer_1 = []

        # 更新上层网络
        ind_last = GL.get_value(ind_thread, "ind_micro")
        # if stepsInOneEp % UPDATE_ITER_HIGH == 0 or is_done:
        if ind_last == -99 or ind_last == 666:
          learning_rate_a_high = FLAGS.learning_rate * (1 - 0.9 * counter / FLAGS.max_steps)  # 根据当前进行完的回合数量修改学习速率（减小）
          learning_rate_c_high = FLAGS.learning_rate * (1 - 0.9 * counter / FLAGS.max_steps)  # 根据当前进行完的回合数量修改学习速率（减小）
          agent.update_high(ind_thread, replay_buffer_2, dir_high_buffer, FLAGS.discount, learning_rate_a_high, learning_rate_c_high, counter)
          # time.sleep(2)
          replay_buffer_2 = []
          dir_high_buffer = []

        if is_done:
          if counter % FLAGS.snapshot_step == 1:    # 到规定回合数存储网络参数（tf.train.Saver().save(),见a3c_agent）
            agent.save_model(SNAPSHOT, counter)
          if counter >= FLAGS.max_steps:    # 超过设定的最大训练回合数后，退出循环（等于线程结束）
            break

      elif is_done:     # 非训练模式下，回合结束的话，根据观测环境的结果在屏幕上打印出得分（is_done成立时的操作对比：
        obs = recorder[-1].observation          # 训练模式下Counter/counter会加一，会更新网络参数，有可能存储网络参数，不打印得分；
        score = obs["score_cumulative"][0]      # 非训练模式下不更新Counter/counter,不更新不存储网络参数，打印得分
        print('Your score is '+str(score)+'!')

    if FLAGS.save_replay:   # 若设置为True，保存该线程的游戏回放
      env.save_replay(agent.name)


def _main(unused_argv):
  """Run agents"""
  stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace   # 应该是开启类似计时时钟这样的观测量
  stopwatch.sw.trace = FLAGS.trace

  maps.get(FLAGS.map)  # Assert the map exists.

  # Setup agents
  agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
  agent_cls = getattr(importlib.import_module(agent_module), agent_name)    # 经过两行操作后，agent_cls就相当于A3CAgent类了，可用其构造对象

  agents = []
  for i in range(PARALLEL):
    agent = agent_cls(FLAGS.training, FLAGS.minimap_resolution, FLAGS.screen_resolution)    # 用agent_cls(A3CAgent)构造对象（调用了a3c_agent文件__init__构造函数）
    agent.build_model(i > 0, DEVICE[i % len(DEVICE)], FLAGS.net)    # 【现在agent就是一个被生成好的A3CAgent对象了】，利用build_model创建所有需要的tf节点
    agents.append(agent)    # agents是多个A3CAgent对象合集（如果PARALLEL大于1的话，不然就只有一个对象在里面）

  config = tf.ConfigProto(allow_soft_placement=True)    # 允许tf自动选择一个存在并且可用的设备来运行操作
  config.gpu_options.allow_growth = True                # 动态申请显存
  sess = tf.Session(config=config)

  summary_writer = tf.summary.FileWriter(LOG)       # 记录日志，供tensorboard使用
  for i in range(PARALLEL):
    agents[i].setup(sess, summary_writer)   # setup各个agent，即 将唯一的sess和summary_writer赋予每个agent的进程
    # print(agents[i])

  # agent就是“包含agents.append(agent)的循环”当中的那个局部变量agent，局部变量能够在外部使用，大概是python(3)神奇的特性...所以agent就是最后一个创建的A3CAgent对象
  agent.initialize()    # run(tf.global_variables_initializer())以初始化每个agent中的tf图
  # print('agent is ==== ', agent)

  if not FLAGS.training or FLAGS.continuation:  # 若不是训练模式 或 若是持续性训练，则利用原有数据（训练好的参数，存在了snapshot文件夹里）进行训练
    global COUNTER
    COUNTER = agent.load_model(SNAPSHOT)    # 全局变量COUNTER记录的是当前所有线程加在一起，总共完成的回合数

  # Run threads
  threads = []
  for i in range(PARALLEL - 1):  # 建立PARALLEL - 1个线程并运行
    t = threading.Thread(target=run_thread, args=(agents[i], FLAGS.map, False, i))     # threading是python自己的线程模块，参数1为线程运行的函数名称，参数2为该函数需要的参数
    threads.append(t)
    t.daemon = True
    t.start()
    time.sleep(5)

  run_thread(agents[-1], FLAGS.map, FLAGS.render, PARALLEL-1)
  # 序号-1代表最后一个agent 这个线程运行时将可视化feature map（因为最后一个参数FLAGS.render为True，之前的几个线程改参数为False）

  for t in threads:
    t.join()    # 必须写 才能正常运行多线程

  if FLAGS.profile:
    print(stopwatch.sw)


if __name__ == "__main__":
  app.run(_main)

