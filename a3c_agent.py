from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from pysc2.lib import actions
from pysc2.lib import features
# DHN add:
from a3c_network import build_high_net
from a3c_network import build_low_net
from high_reward import high_reward
from low_reward import low_reward
import random
import globalvar as GL
import utils as U

# DHN add:
_, num_macro_action = GL.get_list()
spatial_size = 4096


# 在main.py中调用build_model和initialize，run_thread函数按需调用update_low和high，run_loop函数调用step_low和high

class A3CAgent(object):
    """An agent specifically for solving the mini-game maps."""

    def __init__(self, training, msize, ssize, name='A3C/A3CAgent'):
        '''[summary]
        参数为
        是否为训练模式（bool值）
        minimap分辨率（64，整型）
        screen分辨率（64，整型）
        '''

        self.name = name
        self.training = training
        self.summary_low = []
        self.summary_high = []
        self.reward_sum_decay = 0.99  # 与config的discount同步，用于计算reward_decay，动态评估与改变学习率和epsilon-greedy
        # Minimap size, screen size and info size
        assert msize == ssize
        self.msize = msize
        self.ssize = ssize
        # self.isize = len(actions.FUNCTIONS)
        self.info_size_high = 10
        # 当前step，矿物，闲置农民，剩余人口，农民数量，军队数量，房子数量，兵营数量，击杀单位奖励，击杀建筑奖励
        self.info_size_low = 7
        # 当前step，房子数量，兵营数量，击杀单位奖励，击杀建筑奖励，dir_high，act_id

    def setup(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer

    def initialize(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def reset(self):
        # Epsilon schedule
        self.epsilon = [0.05, 0.2]

    def build_model(self, reuse, dev):
        with tf.variable_scope(self.name) and tf.device(dev):
            if reuse:
                # 比如训练模式下4线程，除了第一个build_model的reuse是False以外，其他的均为True（main文件 124行）
                tf.get_variable_scope().reuse_variables()
                assert tf.get_variable_scope().reuse

            # Set inputs of networks 网络输入量为以下3项
            self.minimap = tf.placeholder(tf.float32, [None, U.minimap_channel(), self.msize, self.msize],
                                          name='minimap')
            self.screen = tf.placeholder(tf.float32, [None, U.screen_channel(), self.ssize, self.ssize], name='screen')
            # self.info_high = tf.placeholder(tf.float32, [None, self.isize + self.info_plus_size_high], name='info_high')
            # self.info_low = tf.placeholder(tf.float32, [None, self.isize + self.info_plus_size_low], name='info_low')
            self.info_high = tf.placeholder(tf.float32, [None, self.info_size_high], name='info_high')
            self.info_low = tf.placeholder(tf.float32, [None, self.info_size_low], name='info_low')
            # self.dir_high_usedToFeedLowNet = tf.placeholder(tf.float32, [None, 1, 1], name='dir_high_usedToFeedLowNet')
            # self.act_id = tf.placeholder(tf.float32, [None, 1, 1], name='act_id')
            # self.dir_high_usedToFeedLowNet = tf.placeholder(tf.float32, [1, 1], name='dir_high_usedToFeedLowNet')
            # self.act_id = tf.placeholder(tf.float32, [1, 1], name='act_id')

            # Build networks
            # DHN add:
            self.action_high_prob, self.value_high, self.a_params_high, self.c_params_high = build_high_net(
                self.minimap,
                self.screen,
                self.info_high,
                num_macro_action)
            self.action_low_prob, self.value_low, self.a_params_low, self.c_params_low = build_low_net(self.minimap,
                                                                                                       self.screen,
                                                                                                       self.info_low,
                                                                                                       spatial_size)
            # self.dir_high_usedToFeedLowNet
            # self.act_id

            # Set targets and masks
            # value_target是v现实，是算完以后传进来的（219行），和莫烦A3C一致（莫烦A3C中56和154行）
            # DHN add：
            self.valid_spatial_action_low = tf.placeholder(tf.float32, [None], name='valid_spatial_action_low')
            self.spatial_action_selected_low = tf.placeholder(tf.float32, [None, self.ssize ** 2],
                                                              name='spatial_action_selected_low')
            self.value_target_low = tf.placeholder(tf.float32, [None], name='value_target_low')
            self.value_target_high = tf.placeholder(tf.float32, [None], name='value_target_high')
            self.dir_high_selected = tf.placeholder(tf.float32, [None, num_macro_action], name='dir_high_selected')

            # Compute log probability
            # 用法可以参考Matrix_dot-multiply.py
            # spatial_action是网络输出的坐标，维度是“更新时历经的step数” x “ssize**2”
            # spatial_action_selected含义是每一个step需不需要坐标参数（第一维上），且具体坐标参数是什么（第二维上）。spatial_action_selected维度是“更新时历经的step数” x “ssize**2”
            # 维度是“更新时历经的step数”
            action_low_prob = tf.reduce_sum(self.action_low_prob * self.spatial_action_selected_low, axis=1)
            action_low_log_prob = tf.log(tf.clip_by_value(action_low_prob, 1e-10, 1.))

            action_high_prob = tf.reduce_sum(self.action_high_prob * self.dir_high_selected, axis=1)
            action_high_log_prob = tf.log(tf.clip_by_value(action_high_prob, 1e-10, 1.))
            self.summary_low.append(tf.summary.histogram('action_low_prob', action_low_prob))
            self.summary_high.append(tf.summary.histogram('action_high_prob', action_high_prob))

            # Compute losses, more details in https://arxiv.org/abs/1602.01783
            # 计算网络的a、c loss(这里主要参考莫烦a3c的discrete action程序)
            # 下层网络：
            td_low = tf.subtract(self.value_target_low, self.value_low, name='TD_error_low')
            self.c_loss_low = tf.reduce_mean(tf.square(td_low))
            log_prob_low = self.valid_spatial_action_low * action_low_log_prob  # valid_spatial_action含义是每一个step需不需要坐标参数，维度是“更新时历经的step数”
            self.exp_v_low = log_prob_low * tf.stop_gradient(td_low)
            self.a_loss_low = - tf.reduce_mean(self.exp_v_low)  # 这里不需要像莫烦那样添加一步“增加探索度的操作”了，因为在step_low里已经设置了增加探索度的操作

            # 上层网络：
            td_high = tf.subtract(self.value_target_high, self.value_high, name='TD_error_high')
            self.c_loss_high = tf.reduce_mean(tf.square(td_high))
            self.exp_v_high = action_high_log_prob * tf.stop_gradient(td_high)
            self.a_loss_high = - tf.reduce_mean(self.exp_v_high)
            # 这里不需要epsilon greedy增加探索度了（像莫烦那样），因为在step_low里已经设置了增加探索度的操作

            # 添加summary：
            self.summary_low.append(tf.summary.scalar('a_loss_low', self.a_loss_low))
            self.summary_low.append(tf.summary.scalar('c_loss_low', self.c_loss_low))
            self.summary_high.append(tf.summary.scalar('a_loss_high', self.a_loss_high))
            self.summary_high.append(tf.summary.scalar('c_loss_high', self.c_loss_high))

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

    # DHN add:
    def step_high(self, obs, ind_thread):  # obs就是环境传入的timestep
        minimap = np.array(obs.observation['feature_minimap'],
                           dtype=np.float32)  # 以下4行将minimap和screen的特征做一定处理后分别保存在minimap和screen变量中
        minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)  # 这四行具体语法暂未研究
        screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
        screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
        # info = np.zeros([1, self.isize], dtype=np.float32)  # self.isize值是动作函数的数量
        # info[0, obs.observation['available_actions']] = 1  # info存储可执行的动作。
        # info_plus_high: 当前step，矿物，闲置农民，剩余人口，农民数量，军队数量，房子数量，兵营数量，击杀奖励
        step_count = GL.get_value(ind_thread, "num_steps")
        minerals = obs.observation.player.minerals
        idle_worker = obs.observation.player.idle_worker_count
        food_remain = obs.observation["player"][4] - obs.observation["player"][3]
        worker_count = obs.observation["player"][6]
        army_count = obs.observation["player"][5]
        supply_num = GL.get_value(ind_thread, "supply_num")
        barrack_num = GL.get_value(ind_thread, "barrack_num")
        killed_unit_score = obs.observation["score_cumulative"][5]
        killed_structure_score = obs.observation["score_cumulative"][6]
        # info_plus_high = np.zeros([1, self.info_plus_size_high], dtype=np.float32)
        # info_plus_high[0] = step_count, minerals, idle_worker, food_remain, worker_count, army_count, \
        #                     supply_num, barrack_num, killed_unit_score, killed_structure_score
        # # info 现在的size 是 isize + info_plus_size
        # info_high = np.concatenate((info, info_plus_high), axis=1)
        info_high = np.zeros([1, self.info_size_high], dtype=np.float32)
        info_high[0] = step_count, minerals, idle_worker, food_remain, worker_count, army_count, \
                       supply_num, barrack_num, killed_unit_score, killed_structure_score
        feed = {self.minimap: minimap,
                self.screen: screen,
                self.info_high: info_high}
        action_high_prob = self.sess.run(self.action_high_prob, feed_dict=feed)
        # 选择出宏动作的编号/id
        # DHN待处理： 可以将dir_high先根据一定的方法筛选一下（比如宏动作中的硬编码微动作是否在obs.observation['available_actions']中）
        # valid_dir_high = obs.observation['available_actions']
        # dir_high_id = np.argmax(dir_high)  # 获取要执行的宏动作id（从0开始）
        # 获取要执行的宏动作id，用莫凡的代码改进，是policy-based的选择
        dir_high_id = np.random.choice(range(action_high_prob.shape[1]), p=action_high_prob.ravel())
        # Epsilon greedy exploration  # 0.05(epsilon[0])的概率随机选一个宏动作（会覆盖之前的dir_high_id）
        # Epsilon greedy 是在step内部的，在返回动作之前。update接收到的动作是greedy之后的动作。
        if self.training and np.random.rand() < self.epsilon[0]:
            dir_high_id = random.randint(0, num_macro_action - 1)
        # if np.random.rand() < self.epsilon[0]:
        #     dir_high_id = random.randint(0, num_macro_action - 1)

        return dir_high_id

    def step_low(self, ind_thread, obs, dir_high, act_id):
        # obs就是环境传入的timestep
        minimap = np.array(obs.observation['feature_minimap'],
                           dtype=np.float32)  # 以下4行将minimap和screen的特征做一定处理后分别保存在minimap和screen变量中
        minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)  # 这四行具体语法暂未研究
        screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
        screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
        # info = np.zeros([1, self.isize], dtype=np.float32)  # self.isize值是动作函数的数量
        # info[0, obs.observation['available_actions']] = 1  # info存储可执行的动作。
        # info_plus_low: 当前step，房子数量，兵营数量，击杀奖励
        step_count = GL.get_value(ind_thread, "num_steps")
        supply_num = GL.get_value(ind_thread, "supply_num")
        barrack_num = GL.get_value(ind_thread, "barrack_num")
        killed_unit_score = obs.observation["score_cumulative"][5]
        killed_structure_score = obs.observation["score_cumulative"][6]
        # info_plus_low = np.zeros([1, self.info_plus_size_low], dtype=np.float32)
        # info_plus_low[0] = step_count, supply_num, barrack_num, killed_unit_score, killed_structure_score
        # info_low = np.concatenate((info, info_plus_low), axis=1)
        info_low = np.zeros([1, self.info_size_low], dtype=np.float32)
        info_low[0] = step_count, supply_num, barrack_num, killed_unit_score, killed_structure_score, dir_high, act_id
        # dir_high_usedToFeedLowNet = np.ones([1, 1], dtype=np.float32)
        # dir_high_usedToFeedLowNet[0][0] = dir_high
        # act_ID = np.ones([1, 1], dtype=np.float32)
        # act_ID[0][0] = act_id

        feed = {self.minimap: minimap,
                self.screen: screen,
                self.info_low: info_low}
        # self.dir_high_usedToFeedLowNet: dir_high_usedToFeedLowNet,
        # self.act_id: act_ID
        # 数据类型：Tensor("actor_low/Softmax:0", shape=(?, 4096), dtype=float32, device=/device:GPU:0)
        # [array([[0.00019935, 0.00025348, 0.00024519, ..., 0.00016189, 0.00016014, 0.00016842]], dtype=float32)]
        action_low_prob = self.sess.run(self.action_low_prob, feed_dict=feed)

        # 选择施加动作的位置
        # spatial_action_low = spatial_action_low.ravel()  # ravel()是numpy的函数，作用是将数据降维
        # target = np.argmax(spatial_action_low)
        # 获取坐标位置的4096编号值，用莫凡的代码改进，是policy-based的选择
        position_id = np.random.choice(range(action_low_prob.shape[1]), p=action_low_prob.ravel())
        target = [int(position_id // self.ssize),
                  int(position_id % self.ssize)]  # 获取要施加动作的位置 疑问：若action是勾选方框怎么办？target只有一个坐标吧，那另一个坐标呢？
        # Epsilon greedy exploration  # 0.2(epsilon[1])的概率随机选一个位置施加动作
        if self.training and np.random.rand() < self.epsilon[1]:
            # if np.random.rand() < self.epsilon[1]:
            dy = np.random.randint(-4, 5)
            target[0] = int(max(0, min(self.ssize - 1, target[0] + dy)))
            dx = np.random.randint(-4, 5)
            target[1] = int(max(0, min(self.ssize - 1, target[1] + dx)))

        return target[0], target[1]

    def update_low(self, ind_thread, rbs, dhs, disc, lr_a, lr_c, cter, macro_type, coord_type):
        # rbs(replayBuffers)是[last_timesteps[0], actions[0], timesteps[0]]的集合（agent在一回合里进行了多少step就有多少个），具体见run_loop25行
        # Compute R, which is value of the last observation
        obs = rbs[-1][-1]  # rbs的最后一个元素，应当是当前一步的timesteps值。即obs可以看作timesteps
        if obs.last():
            R = 0
        else:
            minimap = np.array(obs.observation['feature_minimap'], dtype=np.float32)  # 类似105-111行
            minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
            screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
            screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
            # info = np.zeros([1, self.isize], dtype=np.float32)
            # info[0, obs.observation['available_actions']] = 1
            # info_plus_low: 当前step，房子数量，兵营数量，击杀奖励
            step_count = GL.get_value(ind_thread, "num_steps")
            supply_num = GL.get_value(ind_thread, "supply_num")
            barrack_num = GL.get_value(ind_thread, "barrack_num")
            killed_unit_score = obs.observation["score_cumulative"][5]
            killed_structure_score = obs.observation["score_cumulative"][6]
            # info_plus_low = np.zeros([1, self.info_plus_size_low], dtype=np.float32)
            # info_plus_low[0] = step_count, supply_num, barrack_num, killed_unit_score, killed_structure_score
            # info_low = np.concatenate((info, info_plus_low), axis=1)
            info_low = np.zeros([1, self.info_size_low], dtype=np.float32)
            dir_high = dhs[0]
            act_id = GL.get_value(ind_thread, "act_id_micro")
            info_low[
                0] = step_count, supply_num, barrack_num, killed_unit_score, killed_structure_score, dir_high, act_id
            # dir_high_usedToFeedLowNet = np.ones([1, 1], dtype=np.float32)
            # dir_high_usedToFeedLowNet[0][0] = dhs[0]
            # act_id = np.ones([1, 1], dtype=np.float32)
            # act_ID[0][0] = rbs[-1][1].function
            # 之所以不能用rbs里的action信息，是因为rbs里的action可能是no_op(由于出现动作not valid/不合法的情况，为了使游戏不崩掉而不得不这么办的补救措施)
            # 但这里要输入的act_id应该是step_low算出来的act_id
            # act_id[0][0] = GL.get_value(ind_thread, "act_id_micro")

            feed = {self.minimap: minimap,
                    self.screen: screen,
                    self.info_low: info_low}
            # self.dir_high_usedToFeedLowNet: dir_high_usedToFeedLowNet,
            # self.act_id: act_id,
            R = self.sess.run(self.value_low, feed_dict=feed)[0]

        # Compute targets and masks
        minimaps = []
        screens = []
        infos = []
        # dir_highs = []
        # act_ids = []

        value_target = np.zeros([len(rbs)], dtype=np.float32)  # len(rbs) 计算出agent在回合里总共进行的步数
        value_target[-1] = R
        valid_spatial_action = np.zeros([len(rbs)], dtype=np.float32)  # 含义是每一个step需不需要坐标参数
        spatial_action_selected = np.zeros([len(rbs), self.ssize ** 2],
                                           dtype=np.float32)  # 含义是每一个step需不需要坐标参数（第一维上），且具体坐标参数是什么（第二维上）

        rbs.reverse()  # 先reverse 与莫烦A3C_continuous_action.py的代码类似
        micro_isdone = GL.get_value(ind_thread, "micro_isdone")
        micro_isdone.reverse()
        sum_low_reward = GL.get_value(ind_thread, "sum_low_reward")
        for i, [obs, action, next_obs] in enumerate(rbs):  # agent在回合里进行了多少步，就进行多少轮循环
            minimap = np.array(obs.observation['feature_minimap'], dtype=np.float32)  # 类似105-111行
            minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
            screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
            screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
            step_count = GL.get_value(ind_thread, "num_steps")
            supply_num = GL.get_value(ind_thread, "supply_num")
            barrack_num = GL.get_value(ind_thread, "barrack_num")
            killed_unit_score = obs.observation["score_cumulative"][5]
            killed_structure_score = obs.observation["score_cumulative"][6]
            info_low = np.zeros([1, self.info_size_low], dtype=np.float32)
            dir_high = dhs[i]
            act_id = GL.get_value(ind_thread, "act_id_micro")
            info_low[
                0] = step_count, supply_num, barrack_num, killed_unit_score, killed_structure_score, dir_high, act_id
            # dir_high_usedToFeedLowNet = np.ones([1, 1], dtype=np.float32)
            # dir_high_usedToFeedLowNet[0][0] = dhs[i]
            # act_ID = np.ones([1, 1], dtype=np.float32)
            # act_ID[0][0] = GL.get_value(ind_thread, "act_id_micro")
            minimaps.append(minimap)
            screens.append(screen)
            infos.append(info_low)
            # dir_highs.append(dir_high_usedToFeedLowNet)
            # act_ids.append(act_ID)
            # dir_highs.append(dir_high_usedToFeedLowNet)
            # act_ids.append(act_ID)
            coord = [0, 0]
            # coord[0], coord[1] = [32, 32]
            coord[0], coord[1] = self.step_low(ind_thread, obs, dir_high, act_id)
            reward = low_reward(next_obs, obs, coord, macro_type, coord_type, ind_thread)
            sum_low_reward += reward
            GL.add_value_list(ind_thread, "low_reward_of_episode", reward)
            act_id = action.function  # Agent在这一步中选择动作的id序号
            act_args = action.arguments
            value_target[i] = reward + disc * value_target[i - 1]
            # 可参考莫烦Q_Learning教程中对Gamma的意义理解的那张图（有3个眼镜那张），得到回合中每个状态的价值V_S
            # 这里没像莫烦一样再次reverse value 似乎是因为其他参数（如minimap、screen、info等）也都是最后往前反序排列的。见181-182行
            args = actions.FUNCTIONS[act_id].args
            for arg, act_arg in zip(args, act_args):
                if arg.name in ('screen', 'minimap', 'screen2'):
                    ind = act_arg[1] * self.ssize + act_arg[0]
                    valid_spatial_action[i] = 1
                    spatial_action_selected[i, ind] = 1

        GL.set_value(ind_thread, "sum_low_reward", sum_low_reward)
        minimaps = np.concatenate(minimaps, axis=0)
        screens = np.concatenate(screens, axis=0)
        infos = np.concatenate(infos, axis=0)
        # 实际上由于low_net是单步更新策略，所以以下feed的参数里面都只有一帧的数据
        # Train
        feed = {self.minimap: minimaps,
                self.screen: screens,
                self.info_low: infos,
                # self.dir_high_usedToFeedLowNet: dir_highs,
                # self.act_id: act_ids,
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
        dir_high_selected = np.zeros([len(rbs), num_macro_action],
                                     dtype=np.float32)  # 含义是每一个step需不需要坐标参数（第一维上），且具体坐标参数是什么（第二维上）
        for i in range(len(rbs)):
            dir_high_selected[i, dhs[i][0] - 1] = 1
        # Compute R, which is value of the last observation
        obs = rbs[-1][-1]  # rbs的最后一个元素，应当是当前一步的timesteps值。即obs可以看作timesteps
        if obs.last():
            R = 0
        else:
            minimap = np.array(obs.observation['feature_minimap'], dtype=np.float32)  # 类似105-111行
            minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
            screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
            screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
            # info = np.zeros([1, self.isize], dtype=np.float32)
            # info[0, obs.observation['available_actions']] = 1
            # info_plus_high: 当前step，矿物，闲置农民，剩余人口，农民数量，军队数量，房子数量，兵营数量，击杀奖励
            step_count = GL.get_value(ind_thread, "num_steps")
            minerals = obs.observation.player.minerals
            idle_worker = obs.observation.player.idle_worker_count
            food_remain = obs.observation["player"][4] - obs.observation["player"][3]
            worker_count = obs.observation["player"][6]
            army_count = obs.observation["player"][5]
            supply_num = GL.get_value(ind_thread, "supply_num")
            barrack_num = GL.get_value(ind_thread, "barrack_num")
            killed_unit_score = obs.observation["score_cumulative"][5]
            killed_structure_score = obs.observation["score_cumulative"][6]
            # info_plus_high = np.zeros([1, self.info_plus_size_high], dtype=np.float32)
            # info_plus_high[0] = step_count, minerals, idle_worker, food_remain, worker_count, army_count, \
            #                     supply_num, barrack_num, killed_unit_score, killed_structure_score
            # info_high = np.concatenate((info, info_plus_high), axis=1)
            info_high = np.zeros([1, self.info_size_high], dtype=np.float32)
            info_high[0] = step_count, minerals, idle_worker, food_remain, worker_count, army_count, \
                           supply_num, barrack_num, killed_unit_score, killed_structure_score
            feed = {self.minimap: minimap,
                    self.screen: screen,
                    self.info_high: info_high}
            R = self.sess.run(self.value_high, feed_dict=feed)[0]
        # Compute targets and masks
        minimaps = []
        screens = []
        infos = []
        value_target = np.zeros([len(rbs)], dtype=np.float32)  # len(rbs) 计算出agent在回合里总共进行的步数
        value_target[-1] = R
        valid_spatial_action = np.zeros([len(rbs)], dtype=np.float32)  # 含义是每一个step需不需要坐标参数
        spatial_action_selected = np.zeros([len(rbs), self.ssize ** 2],
                                           dtype=np.float32)  # 含义是每一个step需不需要坐标参数（第一维上），且具体坐标参数是什么（第二维上）
        rbs.reverse()  # 先reverse 与莫烦A3C_continuous_action.py的代码类似
        micro_isdone = GL.get_value(ind_thread, "micro_isdone")
        micro_isdone.reverse()
        sum_high_reward = GL.get_value(ind_thread, "sum_high_reward")
        for i, [obs, action, next_obs] in enumerate(rbs):  # agent在回合里进行了多少步，就进行多少轮循环
            minimap = np.array(obs.observation['feature_minimap'], dtype=np.float32)  # 类似105-111行
            minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
            screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
            screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
            # info = np.zeros([1, self.isize], dtype=np.float32)
            # info[0, obs.observation['available_actions']] = 1
            # info_plus_high: 当前step，矿物，闲置农民，剩余人口，农民数量，军队数量，房子数量，兵营数量，击杀奖励
            step_count = GL.get_value(ind_thread, "num_steps")
            minerals = obs.observation.player.minerals
            idle_worker = obs.observation.player.idle_worker_count
            food_remain = obs.observation["player"][4] - obs.observation["player"][3]
            worker_count = obs.observation["player"][6]
            army_count = obs.observation["player"][5]
            supply_num = GL.get_value(ind_thread, "supply_num")
            barrack_num = GL.get_value(ind_thread, "barrack_num")
            killed_unit_score = obs.observation["score_cumulative"][5]
            killed_structure_score = obs.observation["score_cumulative"][6]
            # info_plus_high = np.zeros([1, self.info_plus_size_high], dtype=np.float32)
            # info_plus_high[0] = step_count, minerals, idle_worker, food_remain, worker_count, army_count, \
            #                     supply_num, barrack_num, killed_unit_score, killed_structure_score
            # info_high = np.concatenate((info, info_plus_high), axis=1)
            info_high = np.zeros([1, self.info_size_high], dtype=np.float32)
            info_high[0] = step_count, minerals, idle_worker, food_remain, worker_count, army_count, \
                           supply_num, barrack_num, killed_unit_score, killed_structure_score

            minimaps.append(minimap)
            screens.append(screen)
            infos.append(info_high)
            reward = high_reward(ind_thread, next_obs, obs, action, micro_isdone[i])  # 翔森设计的high reward
            sum_high_reward += reward
            GL.add_value_list(ind_thread, "high_reward_of_episode", reward)
            act_id = action.function  # Agent在这一步中选择动作的id序号
            act_args = action.arguments
            value_target[i] = reward + disc * value_target[i - 1]
            # 可参考莫烦Q_Learning教程中对Gamma的意义理解的那张图（有3个眼镜那张），得到回合中每个状态的价值V_S
            # 这里没像莫烦一样再次reverse value 似乎是因为其他参数（如minimap、screen、info等）也都是最后往前反序排列的。见181-182行
            args = actions.FUNCTIONS[act_id].args
            for arg, act_arg in zip(args, act_args):
                if arg.name in ('screen', 'minimap', 'screen2'):
                    ind = act_arg[1] * self.ssize + act_arg[0]
                    valid_spatial_action[i] = 1
                    spatial_action_selected[i, ind] = 1

        GL.set_value(ind_thread, "sum_high_reward", sum_high_reward)
        high_reward_decay = sum_high_reward * self.reward_sum_decay
        GL.set_value(ind_thread, "high_reward_decay", high_reward_decay)
        minimaps = np.concatenate(minimaps, axis=0)
        screens = np.concatenate(screens, axis=0)
        infos = np.concatenate(infos, axis=0)
        # Train
        feed = {self.minimap: minimaps,
                self.screen: screens,
                self.info_high: infos,
                self.value_target_high: value_target,
                self.dir_high_selected: dir_high_selected,
                self.learning_rate_a_high: lr_a,
                self.learning_rate_c_high: lr_c}
        _, __, summary = self.sess.run([self.update_a_high, self.update_c_high, self.summary_op_high], feed_dict=feed)
        self.summary_writer.add_summary(summary, cter)
        GL.set_value(ind_thread, "micro_isdone", [])

    def save_model(self, path, count):
        # GL.set_saving(True)
        self.saver.save(self.sess, path + '/model.ckpt', count)

    def load_model(self, path):
        ckpt = tf.train.get_checkpoint_state(path)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        return int(ckpt.model_checkpoint_path.split('-')[-1])
