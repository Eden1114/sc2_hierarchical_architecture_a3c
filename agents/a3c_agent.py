from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from pysc2.lib import actions
from pysc2.lib import features

from agents.network import build_net

#DHN add:
from agents.network import build_high_net
from agents.network import build_low_net
from rewards.high_reward import high_reward
from rewards.low_reward import low_reward
import random
import agents.globalvar as GL

import utils as U

#DHN add:
_, num_macro_action = GL.get_list()

class A3CAgent(object):
  """An agent specifically for solving the mini-game maps."""
  def __init__(self, training, msize, ssize, name='A3C/A3CAgent'):  # 参数为是否为训练模式（bool值），minimap尺寸（64，整型），screen尺寸（64，整型）
    self.name = name
    self.training = training
    self.summary_low = []
    self.summary_high = []
    # Minimap size, screen size and info size
    assert msize == ssize
    self.msize = msize
    self.ssize = ssize
    self.isize = len(actions.FUNCTIONS)


  def setup(self, sess, summary_writer):
    self.sess = sess
    self.summary_writer = summary_writer


  def initialize(self):
    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)


  def reset(self):
    # Epsilon schedule
    self.epsilon = [0.05, 0.2]


  def build_model(self, reuse, dev, ntype):
    with tf.variable_scope(self.name) and tf.device(dev):
      if reuse:   #比如训练模式下4线程，除了第一个build_model的reuse是False以外，其他的均为True（main文件 124行）
        tf.get_variable_scope().reuse_variables()
        assert tf.get_variable_scope().reuse

      # Set inputs of networks 网络输入量为以下3项
      self.minimap = tf.placeholder(tf.float32, [None, U.minimap_channel(), self.msize, self.msize], name='minimap')
      self.screen = tf.placeholder(tf.float32, [None, U.screen_channel(), self.ssize, self.ssize], name='screen')
      self.info = tf.placeholder(tf.float32, [None, self.isize], name='info')


      # Build networks
      # net = build_net(self.minimap, self.screen, self.info, self.msize, self.ssize, len(actions.FUNCTIONS), ntype)  # build_net函数从network.py中引入
      # self.spatial_action, self.non_spatial_action, self.value = net  # 可见，build_model中建立的3个网络都是同样的结构


      # DHN add:
      self.dir_high, self.value_high, self.a_params_high, self.c_params_high = build_high_net(self.minimap, self.screen, self.info, num_macro_action)
      self.spatial_action_low, self.value_low, self.a_params_low, self.c_params_low = build_low_net(self.minimap, self.screen, self.info)


      # Set targets and masks
      # self.valid_spatial_action = tf.placeholder(tf.float32, [None], name='valid_spatial_action')
      # self.spatial_action_selected = tf.placeholder(tf.float32, [None, self.ssize**2], name='spatial_action_selected')
      # self.valid_non_spatial_action = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)], name='valid_non_spatial_action')
      # self.non_spatial_action_selected = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)], name='non_spatial_action_selected')
      # self.value_target = tf.placeholder(tf.float32, [None], name='value_target')   #value_target是v现实，是算完以后传进来的（219行），和莫烦A3C一致（莫烦A3C中56和154行）


      #DHN add：
      self.valid_spatial_action_low = tf.placeholder(tf.float32, [None], name='valid_spatial_action_low')
      self.spatial_action_selected_low = tf.placeholder(tf.float32, [None, self.ssize ** 2], name='spatial_action_selected_low')
      self.value_target_low = tf.placeholder(tf.float32, [None], name='value_target_low')
      self.value_target_high = tf.placeholder(tf.float32, [None], name='value_target_high')
      self.dir_high_selected = tf.placeholder(tf.float32, [None, num_macro_action], name='dir_high_selected')


      # Compute log probability
      spatial_action_prob_low = tf.reduce_sum(self.spatial_action_low * self.spatial_action_selected_low, axis=1)   # 用法可以参考Matrix_dot-multiply.py
            # spatial_action是网络输出的坐标，维度是“更新时历经的step数” x “ssize**2”
            # spatial_action_selected含义是每一个step需不需要坐标参数（第一维上），且具体坐标参数是什么（第二维上）。spatial_action_selected维度是“更新时历经的step数” x “ssize**2”
      spatial_action_log_prob_low = tf.log(tf.clip_by_value(spatial_action_prob_low, 1e-10, 1.))  # 维度是“更新时历经的step数”

      dir_prob_high = tf.reduce_sum(self.dir_high * self.dir_high_selected, axis=1)
      dir_log_prob_high = tf.log(tf.clip_by_value(dir_prob_high, 1e-10, 1.))

      # non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.non_spatial_action_selected, axis=1)
      # valid_non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.valid_non_spatial_action, axis=1)
      # valid_non_spatial_action_prob = tf.clip_by_value(valid_non_spatial_action_prob, 1e-10, 1.)
      # non_spatial_action_prob = non_spatial_action_prob / valid_non_spatial_action_prob
      # non_spatial_action_log_prob = tf.log(tf.clip_by_value(non_spatial_action_prob, 1e-10, 1.))

      self.summary_low.append(tf.summary.histogram('spatial_action_prob_low', spatial_action_prob_low))
      self.summary_high.append(tf.summary.histogram('dir_prob_high', dir_prob_high))
      # self.summary.append(tf.summary.histogram('non_spatial_action_prob', non_spatial_action_prob))

      # Compute losses, more details in https://arxiv.org/abs/1602.01783

      # 计算网络的a、c loss(这里主要参考莫烦a3c的discrete action程序)
      # 下层网络：
      td_low = tf.subtract(self.value_target_low, self.value_low, name='TD_error_low')
      self.c_loss_low = tf.reduce_mean(tf.square(td_low))

      log_prob_low = self.valid_spatial_action_low * spatial_action_log_prob_low  # valid_spatial_action含义是每一个step需不需要坐标参数，维度是“更新时历经的step数”
      self.exp_v_low = log_prob_low * tf.stop_gradient(td_low)
      self.a_loss_low = - tf.reduce_mean(self.exp_v_low)     # 这里不需要像莫烦那样添加一步“增加探索度的操作”了，因为在step_low里已经设置了增加探索度的操作

      # 上层网络：
      td_high = tf.subtract(self.value_target_high, self.value_high, name='TD_error_low')
      self.c_loss_high = tf.reduce_mean(tf.square(td_high))

      self.exp_v_high = dir_log_prob_high * tf.stop_gradient(td_high)
      self.a_loss_high = - tf.reduce_mean(self.exp_v_high)  # 这里不需要epsilon greedy增加探索度了（像莫烦那样），因为在step_low里已经设置了增加探索度的操作


      # 添加summary：
      self.summary_low.append(tf.summary.scalar('a_loss_low', self.a_loss_low))
      self.summary_low.append(tf.summary.scalar('c_loss_low', self.c_loss_low))
      self.summary_high.append(tf.summary.scalar('a_loss_high', self.a_loss_high))
      self.summary_high.append(tf.summary.scalar('c_loss_high', self.c_loss_high))

      # TODO: policy penalty
      # loss = policy_loss + value_loss
      # Build the optimizer
      # self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
      # opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, epsilon=1e-10)
      # grads = opt.compute_gradients(loss)
      # cliped_grad = []
      # for grad, var in grads:
      #   self.summary.append(tf.summary.histogram(var.op.name, var))
      #   self.summary.append(tf.summary.histogram(var.op.name+'/grad', grad))
      #   grad = tf.clip_by_norm(grad, 10.0)
      #   cliped_grad.append([grad, var])
      # self.train_op = opt.apply_gradients(cliped_grad)


      # 根据梯度进行更新(这里主要参考莫烦a3c的continuous action程序)
      # 下层网络：
      self.learning_rate_a_low = tf.placeholder(tf.float32, None, name='learning_rate_a_low')
      opt_a_low = tf.train.RMSPropOptimizer(self.learning_rate_a_low, decay=0.99, epsilon=1e-10)
      self.a_grads_low = tf.gradients(self.a_loss_low, self.a_params_low)
      self.update_a_low = opt_a_low.apply_gradients(zip(self.a_grads_low, self.a_params_low))

      self.learning_rate_c_low = tf.placeholder(tf.float32, None, name='learning_rate_c_low')
      opt_c_low = tf.train.RMSPropOptimizer(self.learning_rate_c_low, decay=0.99, epsilon=1e-10)
      self.c_grads_low = tf.gradients(self.c_loss_low, self.c_params_low)
      self.update_c_low = opt_c_low.apply_gradients(zip(self.c_grads_low, self.c_params_low))

      # 上层网络：
      self.learning_rate_a_high = tf.placeholder(tf.float32, None, name='learning_rate_a_high')
      opt_a_high = tf.train.RMSPropOptimizer(self.learning_rate_a_high, decay=0.99, epsilon=1e-10)
      self.a_grads_high = tf.gradients(self.a_loss_high, self.a_params_high)
      self.update_a_high = opt_a_high.apply_gradients(zip(self.a_grads_high, self.a_params_high))

      self.learning_rate_c_high = tf.placeholder(tf.float32, None, name='learning_rate_c_high')
      opt_c_high = tf.train.RMSPropOptimizer(self.learning_rate_c_high, decay=0.99, epsilon=1e-10)
      self.c_grads_high = tf.gradients(self.c_loss_high, self.c_params_high)
      self.update_c_high = opt_c_high.apply_gradients(zip(self.c_grads_high, self.c_params_high))


      self.summary_op_low = tf.summary.merge(self.summary_low)
      self.summary_op_high = tf.summary.merge(self.summary_high)
      self.saver = tf.train.Saver(max_to_keep=100)  # 定义self.saver 为 tf的存储器Saver()，在save_model和load_model函数里使用


# 该函数不再使用：
  def step(self, obs):  # obs就是环境传入的timestep
    minimap = np.array(obs.observation['feature_minimap'], dtype=np.float32)  # 以下4行将minimap和screen的特征做一定处理后分别保存在minimap和screen变量中
    minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)         # 这四行具体语法暂未研究
    screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
    screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
    # TODO: only use available actions
    info = np.zeros([1, self.isize], dtype=np.float32)  # self.isize值是动作函数的数量
    info[0, obs.observation['available_actions']] = 1   # info存储可执行的动作。

    feed = {self.minimap: minimap,
            self.screen: screen,
            self.info: info}
    non_spatial_action, spatial_action = self.sess.run(   # non_spatial_action和spatial_action是两个网络，
      [self.non_spatial_action, self.spatial_action],   # 一个计算非空间动作（大概就是要进行什么操作，如勾选方框、建补给库），一个计算空间内的动作（应该是在空间的哪个位置施加操作）
      feed_dict=feed)

    # Select an action and a spatial target
    non_spatial_action = non_spatial_action.ravel()   # non_spatial_action和spatial_action是两个网络返回的动作值
    # print('non_spatial_action=========', non_spatial_action)
    spatial_action = spatial_action.ravel()           # ravel()是numpy的函数，作用是将数据降维
    valid_actions = obs.observation['available_actions']
    # print('valid_actions=========', valid_actions)
    act_id = valid_actions[np.argmax(non_spatial_action[valid_actions])]  # 获取要执行的动作id
    target = np.argmax(spatial_action)
    target = [int(target // self.ssize), int(target % self.ssize)]  # 获取要施加动作的位置 疑问：若action是勾选方框怎么办？target只有一个坐标吧，那另一个坐标呢？

    if False:   # 疑问：if False什么意思？网上没查到
      print(actions.FUNCTIONS[act_id].name, target)

    # Epsilon greedy exploration  # 0.05(epsilon[0])的概率随机选一个动作（会覆盖之前的act_id），0.2(epsilon[1])的概率随机选一个位置施加动作
    if self.training and np.random.rand() < self.epsilon[0]:  # epsilon值在40行
      act_id = np.random.choice(valid_actions)
    if self.training and np.random.rand() < self.epsilon[1]:
      dy = np.random.randint(-4, 5)
      target[0] = int(max(0, min(self.ssize-1, target[0]+dy)))
      dx = np.random.randint(-4, 5)
      target[1] = int(max(0, min(self.ssize-1, target[1]+dx)))

    # Set act_id and act_args
    act_args = []
    for arg in actions.FUNCTIONS[act_id].args:  # actions是pysc2.lib中的文件 根据act_id获取其可使用的参数，并添加到args中去
      if arg.name in ('screen', 'minimap', 'screen2'):
        act_args.append([target[1], target[0]])
      else:
        act_args.append([0])  # TODO: Be careful
    return actions.FunctionCall(act_id, act_args)

  # DHN add:
  def step_high(self, obs):  # obs就是环境传入的timestep
    minimap = np.array(obs.observation['feature_minimap'], dtype=np.float32)  # 以下4行将minimap和screen的特征做一定处理后分别保存在minimap和screen变量中
    minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)         # 这四行具体语法暂未研究
    screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
    screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
    # TODO: only use available actions
    info = np.zeros([1, self.isize], dtype=np.float32)  # self.isize值是动作函数的数量
    info[0, obs.observation['available_actions']] = 1   # info存储可执行的动作。

    feed = {self.minimap: minimap,
            self.screen: screen,
            self.info: info}
    dir_high = self.sess.run(
      [self.dir_high],
      feed_dict=feed)

    # 选择出宏动作的编号/id

    #DHN待处理： 可以将dir_high先根据一定的方法筛选一下（比如宏动作中的硬编码微动作是否在obs.observation['available_actions']中）
    # valid_dir_high = obs.observation['available_actions']

    dir_high_id = np.argmax(dir_high)  # 获取要执行的宏动作id（从0开始）

    # if False:   # 疑问：if False什么意思？网上没查到
    #   print(actions.FUNCTIONS[act_id].name, target)

    # Epsilon greedy exploration  # 0.05(epsilon[0])的概率随机选一个宏动作（会覆盖之前的dir_high_id）
    if self.training and np.random.rand() < self.epsilon[0]:
      dir_high_id = random.randint(0,num_macro_action-1)

    return dir_high_id


  def step_low(self, obs):
    # obs就是环境传入的timestep
    minimap = np.array(obs.observation['feature_minimap'], dtype=np.float32)  # 以下4行将minimap和screen的特征做一定处理后分别保存在minimap和screen变量中
    minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)         # 这四行具体语法暂未研究
    screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
    screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
    # TODO: only use available actions
    info = np.zeros([1, self.isize], dtype=np.float32)  # self.isize值是动作函数的数量
    info[0, obs.observation['available_actions']] = 1   # info存储可执行的动作。

    feed = {self.minimap: minimap,
            self.screen: screen,
            self.info: info}
    spatial_action_low = self.sess.run( # 数据类型：Tensor("actor_low/Softmax:0", shape=(?, 4096), dtype=float32, device=/device:GPU:0)
                                        # [array([[0.00019935, 0.00025348, 0.00024519, ..., 0.00016189, 0.00016014, 0.00016842]], dtype=float32)]
      [self.spatial_action_low],
      feed_dict=feed)

    # 选择施加动作的位置
    # spatial_action_low = spatial_action_low.ravel()  # ravel()是numpy的函数，作用是将数据降维
    target = np.argmax(spatial_action_low)
    target = [int(target // self.ssize), int(target % self.ssize)]  # 获取要施加动作的位置 疑问：若action是勾选方框怎么办？target只有一个坐标吧，那另一个坐标呢？

    # if False:   # 疑问：if False什么意思？网上没查到
    #   print(actions.FUNCTIONS[act_id].name, target)

    # Epsilon greedy exploration  # 0.2(epsilon[1])的概率随机选一个位置施加动作
    if self.training and np.random.rand() < self.epsilon[1]:
      dy = np.random.randint(-4, 5)
      target[0] = int(max(0, min(self.ssize-1, target[0]+dy)))
      dx = np.random.randint(-4, 5)
      target[1] = int(max(0, min(self.ssize-1, target[1]+dx)))

    #DHN待处理：
    # act_id = 1  # 暂时指定为1，之后替换为硬编码的动作id
    # 设置动作函数所需的act_args
    # act_args = []
    # for arg in actions.FUNCTIONS[act_id].args:  # actions是pysc2.lib中的文件 根据act_id获取其可使用的参数，并添加到args中去
    #   if arg.name in ('screen', 'minimap', 'screen2'):
    #     act_args.append([target[1], target[0]])
    #   else:
    #     act_args.append([0])  # TODO: Be careful
    # return actions.FunctionCall(act_id, act_args)
    return target[0], target[1]

# 该函数不再使用：
  def update(self, rbs, disc, lr, cter):  # rbs是[last_timesteps[0], actions[0], timesteps[0]]的集合（agent在一回合里进行了多少step就有多少个），具体见run_loop25行
    # Compute R, which is value of the last observation
    obs = rbs[-1][-1]   # rbs的最后一个元素，应当是当前一步的timesteps值。即obs可以看作timesteps
    if obs.last():
      R = 0
    else:
      minimap = np.array(obs.observation['feature_minimap'], dtype=np.float32)  # 类似105-111行
      minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
      screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
      screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
      info = np.zeros([1, self.isize], dtype=np.float32)
      info[0, obs.observation['available_actions']] = 1

      feed = {self.minimap: minimap,
              self.screen: screen,
              self.info: info}
      R = self.sess.run(self.value, feed_dict=feed)[0]

    # Compute targets and masks
    minimaps = []
    screens = []
    infos = []

    value_target = np.zeros([len(rbs)], dtype=np.float32)   # len(rbs) 计算出agent在回合里总共进行的步数
    value_target[-1] = R

    valid_spatial_action = np.zeros([len(rbs)], dtype=np.float32)       # 含义是每一个step需不需要坐标参数
    spatial_action_selected = np.zeros([len(rbs), self.ssize**2], dtype=np.float32) # 含义是每一个step需不需要坐标参数（第一维上），且具体坐标参数是什么（第二维上）
    valid_non_spatial_action = np.zeros([len(rbs), len(actions.FUNCTIONS)], dtype=np.float32)
    non_spatial_action_selected = np.zeros([len(rbs), len(actions.FUNCTIONS)], dtype=np.float32)

    rbs.reverse()  # 先reverse，再进行198行的操作，与莫烦A3C_continuous_action.py 145行开始的代码类似
    for i, [obs, action, next_obs] in enumerate(rbs):   # agent在回合里进行了多少步，就进行多少轮循环
      minimap = np.array(obs.observation['feature_minimap'], dtype=np.float32)  # 类似105-111行
      minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
      screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
      screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
      info = np.zeros([1, self.isize], dtype=np.float32)
      info[0, obs.observation['available_actions']] = 1

      minimaps.append(minimap)
      screens.append(screen)
      infos.append(info)

      reward = obs.reward
      act_id = action.function  # Agent在这一步中选择动作的id序号
      act_args = action.arguments

      value_target[i] = reward + disc * value_target[i-1]   # 可参考莫烦Q_Learning教程中对Gamma的意义理解的那张图（有3个眼镜那张），得到回合中每个状态的价值V_S
      # 这里没像莫烦一样再次reverse value 似乎是因为其他参数（如minimap、screen、info等）也都是最后往前反序排列的。见181-182行
      valid_actions = obs.observation["available_actions"]  # valid_actions是个元素数为541的列表，many-hot（参考one-hot进行理解）的列表
      valid_non_spatial_action[i, valid_actions] = 1
      non_spatial_action_selected[i, act_id] = 1

      args = actions.FUNCTIONS[act_id].args
      for arg, act_arg in zip(args, act_args):
        if arg.name in ('screen', 'minimap', 'screen2'):
          ind = act_arg[1] * self.ssize + act_arg[0]
          valid_spatial_action[i] = 1
          spatial_action_selected[i, ind] = 1

    minimaps = np.concatenate(minimaps, axis=0)
    screens = np.concatenate(screens, axis=0)
    infos = np.concatenate(infos, axis=0)

    # Train
    feed = {self.minimap: minimaps,
            self.screen: screens,
            self.info: infos,
            self.value_target: value_target,
            self.valid_spatial_action: valid_spatial_action,
            self.spatial_action_selected: spatial_action_selected,
            self.valid_non_spatial_action: valid_non_spatial_action,
            self.non_spatial_action_selected: non_spatial_action_selected,
            self.learning_rate: lr}
    _, summary = self.sess.run([self.train_op, self.summary_op], feed_dict=feed)  # 关键语句：训练的是self.train_op, self.summary_op（在上面的build_model函数里去查看它们具体是什么）
    self.summary_writer.add_summary(summary, cter)


  def update_low(self, rbs, disc, lr_a, lr_c, cter):
    # rbs(replayBuffers)是[last_timesteps[0], actions[0], timesteps[0]]的集合（agent在一回合里进行了多少step就有多少个），具体见run_loop25行

    # Compute R, which is value of the last observation
    obs = rbs[-1][-1]   # rbs的最后一个元素，应当是当前一步的timesteps值。即obs可以看作timesteps
    if obs.last():
      R = 0
    else:
      minimap = np.array(obs.observation['feature_minimap'], dtype=np.float32)  # 类似105-111行
      minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
      screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
      screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
      info = np.zeros([1, self.isize], dtype=np.float32)
      info[0, obs.observation['available_actions']] = 1

      feed = {self.minimap: minimap,
              self.screen: screen,
              self.info: info}
      R = self.sess.run(self.value_low, feed_dict=feed)[0]

    # Compute targets and masks
    minimaps = []
    screens = []
    infos = []

    value_target = np.zeros([len(rbs)], dtype=np.float32)   # len(rbs) 计算出agent在回合里总共进行的步数
    value_target[-1] = R

    valid_spatial_action = np.zeros([len(rbs)], dtype=np.float32)       # 含义是每一个step需不需要坐标参数
    spatial_action_selected = np.zeros([len(rbs), self.ssize**2], dtype=np.float32) # 含义是每一个step需不需要坐标参数（第一维上），且具体坐标参数是什么（第二维上）

    rbs.reverse()  # 先reverse 与莫烦A3C_continuous_action.py的代码类似
    for i, [obs, action, next_obs] in enumerate(rbs):   # agent在回合里进行了多少步，就进行多少轮循环
      minimap = np.array(obs.observation['feature_minimap'], dtype=np.float32)  # 类似105-111行
      minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
      screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
      screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
      info = np.zeros([1, self.isize], dtype=np.float32)
      info[0, obs.observation['available_actions']] = 1

      minimaps.append(minimap)
      screens.append(screen)
      infos.append(info)

      reward = obs.reward
      act_id = action.function  # Agent在这一步中选择动作的id序号
      act_args = action.arguments

      value_target[i] = reward + disc * value_target[i-1]   # 可参考莫烦Q_Learning教程中对Gamma的意义理解的那张图（有3个眼镜那张），得到回合中每个状态的价值V_S
      # 这里没像莫烦一样再次reverse value 似乎是因为其他参数（如minimap、screen、info等）也都是最后往前反序排列的。见181-182行

      args = actions.FUNCTIONS[act_id].args
      for arg, act_arg in zip(args, act_args):
        if arg.name in ('screen', 'minimap', 'screen2'):
          ind = act_arg[1] * self.ssize + act_arg[0]
          valid_spatial_action[i] = 1
          spatial_action_selected[i, ind] = 1

    minimaps = np.concatenate(minimaps, axis=0)
    screens = np.concatenate(screens, axis=0)
    infos = np.concatenate(infos, axis=0)

    # Train
    feed = {self.minimap: minimaps,
            self.screen: screens,
            self.info: infos,
            self.value_target_low: value_target,
            self.valid_spatial_action_low: valid_spatial_action,
            self.spatial_action_selected_low: spatial_action_selected,
            self.learning_rate_a_low: lr_a,
            self.learning_rate_c_low: lr_c}
    _, __, summary = self.sess.run([self.update_a_low, self.update_c_low, self.summary_op_low], feed_dict=feed)
    self.summary_writer.add_summary(summary, cter)


  def update_high(self, ind_thread, rbs, dhs, disc, lr_a, lr_c, cter):
    # rbs(replayBuffers)是[last_timesteps[0], actions[0], timesteps[0]]的集合（更新时经历了多少个step就有多少个），具体见run_loop25行
    # dhs(dir_high_buffers) 是指令序号的集合。比如一共有5个宏动作，则dhs形如[5, 4, 1, 2, 3, 4, 2, 1, ......]

    dir_high_selected = np.zeros([len(rbs), num_macro_action], dtype=np.float32)  # 含义是每一个step需不需要坐标参数（第一维上），且具体坐标参数是什么（第二维上）
    for i in range(len(rbs)):
      dir_high_selected[i, dhs[i][0]-1] = 1

    # Compute R, which is value of the last observation
    obs = rbs[-1][-1]   # rbs的最后一个元素，应当是当前一步的timesteps值。即obs可以看作timesteps
    if obs.last():
      R = 0
    else:
      minimap = np.array(obs.observation['feature_minimap'], dtype=np.float32)  # 类似105-111行
      minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
      screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
      screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
      info = np.zeros([1, self.isize], dtype=np.float32)
      info[0, obs.observation['available_actions']] = 1

      feed = {self.minimap: minimap,
              self.screen: screen,
              self.info: info}
      R = self.sess.run(self.value_high, feed_dict=feed)[0]

    # Compute targets and masks
    minimaps = []
    screens = []
    infos = []

    value_target = np.zeros([len(rbs)], dtype=np.float32)   # len(rbs) 计算出agent在回合里总共进行的步数
    value_target[-1] = R

    valid_spatial_action = np.zeros([len(rbs)], dtype=np.float32)       # 含义是每一个step需不需要坐标参数
    spatial_action_selected = np.zeros([len(rbs), self.ssize**2], dtype=np.float32) # 含义是每一个step需不需要坐标参数（第一维上），且具体坐标参数是什么（第二维上）

    rbs.reverse()  # 先reverse 与莫烦A3C_continuous_action.py的代码类似
    micro_isdone = GL.get_value(ind_thread, "micro_isdone")
    micro_isdone.reverse()
    for i, [obs, action, next_obs] in enumerate(rbs):   # agent在回合里进行了多少步，就进行多少轮循环
      minimap = np.array(obs.observation['feature_minimap'], dtype=np.float32)  # 类似105-111行
      minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
      screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
      screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
      info = np.zeros([1, self.isize], dtype=np.float32)
      info[0, obs.observation['available_actions']] = 1

      minimaps.append(minimap)
      screens.append(screen)
      infos.append(info)

      # reward = obs.reward
      reward = high_reward(ind_thread, next_obs, obs, action, micro_isdone[i])  # 翔森设计的high reward
      act_id = action.function  # Agent在这一步中选择动作的id序号
      act_args = action.arguments

      value_target[i] = reward + disc * value_target[i-1]   # 可参考莫烦Q_Learning教程中对Gamma的意义理解的那张图（有3个眼镜那张），得到回合中每个状态的价值V_S
      # 这里没像莫烦一样再次reverse value 似乎是因为其他参数（如minimap、screen、info等）也都是最后往前反序排列的。见181-182行

      args = actions.FUNCTIONS[act_id].args
      for arg, act_arg in zip(args, act_args):
        if arg.name in ('screen', 'minimap', 'screen2'):
          ind = act_arg[1] * self.ssize + act_arg[0]
          valid_spatial_action[i] = 1
          spatial_action_selected[i, ind] = 1

    minimaps = np.concatenate(minimaps, axis=0)
    screens = np.concatenate(screens, axis=0)
    infos = np.concatenate(infos, axis=0)

    # Train
    feed = {self.minimap: minimaps,
            self.screen: screens,
            self.info: infos,
            self.value_target_high: value_target,
            self.dir_high_selected: dir_high_selected,
            self.learning_rate_a_high: lr_a,
            self.learning_rate_c_high: lr_c}
    _, __, summary = self.sess.run([self.update_a_high, self.update_c_high, self.summary_op_high], feed_dict=feed)
    self.summary_writer.add_summary(summary, cter)

    GL.set_value(ind_thread, "micro_isdone", [])


  def save_model(self, path, count):
    self.saver.save(self.sess, path+'/model.pkl', count)


  def load_model(self, path):
    ckpt = tf.train.get_checkpoint_state(path)
    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    return int(ckpt.model_checkpoint_path.split('-')[-1])
