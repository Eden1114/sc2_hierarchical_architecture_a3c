# phi，policy hierarchical learning

import numpy as np
from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2 import maps
from pysc2.lib import stopwatch
import tensorflow as tf
import tensorflow.contrib.layers as layers
import globalvar as GL
import a3c_reward as reward
from macro_actions import action_micro
import preprocess as prep


class PHI:
    def __init__(self, sess, macro_action_num, lr, minimap_size, screen_size, info_size_high, info_size_low):
        self.action_num = macro_action_num  # high输出宏动作id，low输出location
        self.sess = sess
        self.learning_rate = lr
        self.minimap_size = minimap_size
        self.screen_size = screen_size
        self.info_size_high = info_size_high
        # 当前step，矿物，闲置农民，剩余人口，农民数量，军队数量，房子数量，兵营数量，击杀单位奖励，击杀建筑奖励
        self.info_size_low = info_size_low
        # 当前step，房子数量，兵营数量，击杀单位奖励，击杀建筑奖励
        # feed_dict = {minimap, screen, info_high, info_low, dir_high, act_id}
        # 下面的[None, ]，应该是为了读取buffer用的。如果我们改变了buffer的定义，这里可以改变
        self.minimap = tf.placeholder(tf.float32, [None, prep.minimap_channel(), self.minimap_size, self.minimap_size])
        self.screen = tf.placeholder(tf.float32, [None, prep.screen_channel(), self.screen_size, self.screen_size])
        # self.info = tf.placeholder(tf.float32, [None, self.action_num])   # 把所有的info（available_actions）都去除了
        self.info_high = tf.placeholder(tf.float32, [None, self.info_size_high])
        self.info_low = tf.placeholder(tf.float32, [None, self.info_size_low])
        self.dir_high = tf.placeholder(tf.float32, [1, 1], name='dir_high')
        self.act_id = tf.placeholder(tf.float32, [1, 1], name='act_id')

        with tf.variable_scope('phi') and tf.device('/gpu:0'):
            # 网络定义
            self.action_high, self.value_high,  self.a_params_high, self.c_params_high = self.build_net_high()
            self.action_low, self.value_low, self.a_params_low, self.c_params_low = self.build_net_low()

            # ———————————————— 网络训练—————————————————— #
            # TODO 需要这部分所使用算法的详细解析
            advantage = tf.stop_gradient(self.q_target_value - self.q_value)

            # 求动作选择的似然概率
            spatial_prob = tf.reduce_sum(self.spatial_action * self.spatial_choose, axis=1)
            spatial_log_prob = tf.log(tf.clip_by_value(spatial_prob, 1e-10, 1.))
            non_spatial_prob = tf.reduce_sum(self.non_spatial_action * self.non_spatial_choose, axis=1)
            valid_non_spatial_prob = tf.reduce_sum(self.non_spatial_action * self.non_spatial_mask, axis=1)
            valid_non_spatial_prob = tf.clip_by_value(valid_non_spatial_prob, 1e-10, 1.)
            non_spatial_prob = non_spatial_prob / valid_non_spatial_prob
            non_spatial_log_prob = tf.log(tf.clip_by_value(non_spatial_prob, 1e-10, 1.))

            action_log_prob = self.spatial_mask * spatial_log_prob + non_spatial_log_prob

            # 策略损失与价值损失
            policy_loss = - tf.reduce_mean(action_log_prob * advantage)
            value_loss = 0.25 * tf.reduce_mean(tf.square(self.q_value * advantage))

            # ——————— 加入entropy loss ——————— #
            # entropy = - (tf.reduce_sum(spatial_prob * spatial_log_prob, axis=-1) +
            #              tf.reduce_sum(non_spatial_prob * non_spatial_log_prob, axis=-1))
            # entropy_loss = - 1e-3 * tf.reduce_mean(entropy)
            # loss = policy_loss + value_loss + entropy_loss

            loss = policy_loss + value_loss

            # ———————————————— 训练定义 —————————————————— #

            opt = tf.train.RMSPropOptimizer(5e-4, decay=0.99, epsilon=1e-10)  # TODO 学习率等参数设置
            grads = opt.compute_gradients(loss)
            cliped_grad = []
            for grad, var in grads:
                grad = tf.clip_by_norm(grad, 10.0)
                cliped_grad.append([grad, var])
            self.train_op = opt.apply_gradients(cliped_grad)
            self.saver = tf.train.Saver(max_to_keep=100)  # 定义self.saver 为 tf的存储器Saver()，在save_model和load_model函数里使用

    def build_net_high(self):
        with tf.variable_scope('network_high'):
            with tf.variable_scope('feature_high'):
                mconv1 = layers.conv2d(tf.transpose(self.minimap, [0, 2, 3, 1]), 16, 5, name='mconv1')
                mconv2 = layers.conv2d(mconv1, 32, 3, name='mconv2')
                sconv1 = layers.conv2d(tf.transpose(self.screen, [0, 2, 3, 1]), 16, 5, name='sconv1')
                sconv2 = layers.conv2d(sconv1, 32, 3, name='sconv2')
                info_high = layers.fully_connected(layers.flatten(self.info_high), 32, activation_fn=None,
                                                   name='info_high')

                full_concat = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2), info_high], axis=1)
                full_feature_high = layers.fully_connected(full_concat, 128, activation_fn=tf.tanh,
                                                           name='full_feature_high')
                # conv_concat = tf.concat([mconv2, sconv2], axis=3)
                # conv_feature_high = layers.conv2d(conv_concat, 1, 1, activation_fn=None, name='conv_feature_high')
            with tf.variable_scope('actor_high'):
                actor_hidden_high = layers.fully_connected(full_feature_high, 32, activation_fn=tf.nn.relu,
                                                           name='actor_hidden_high')
                action_high = layers.fully_connected(actor_hidden_high, self.action_num, activation_fn=tf.nn.softmax,
                                                  name='dir_high')
            with tf.variable_scope('critic_high'):
                critic_hidden_high = layers.fully_connected(full_feature_high, 32, activation_fn=tf.nn.relu,
                                                           name='critic_hidden_high')
                value_high = tf.reshape(layers.fully_connected(critic_hidden_high, 1, activation_fn=tf.tanh, name='value_high'), [-1])

            actor_params_high = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_high')
            critic_params_high = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_high')
            feature_params_high = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='feature_high')
            # 可以在此处重新定义更新参数的方法（考虑为feature_high单独设计梯度）
            a_params_high = actor_params_high + feature_params_high
            c_params_high = critic_params_high + feature_params_high
            print('a_params_high: ', a_params_high)

            return action_high, value_high, a_params_high, c_params_high

    def build_net_low(self):
        with tf.variable_scope('network_low'):
            with tf.variable_scope('feature_low'):
                mconv1 = layers.conv2d(tf.transpose(self.minimap, [0, 2, 3, 1]), 16, 5, name='mconv1')
                mconv2 = layers.conv2d(mconv1, 32, 3, name='mconv2')
                sconv1 = layers.conv2d(tf.transpose(self.screen, [0, 2, 3, 1]), 16, 5, name='sconv1')
                sconv2 = layers.conv2d(sconv1, 32, 3, name='sconv2')
                high_net_output = tf.concat([self.dir_high, self.act_id], axis=1)
                info_concat = tf.concat([layers.flatten(self.info_low), high_net_output], axis=1)
                info_low = layers.fully_connected(info_concat, 32, activation_fn=None,
                                                  name='info_low')

                full_concat = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2), info_low], axis=1)
                full_feature_low = layers.fully_connected(full_concat, 128, activation_fn=tf.tanh,
                                                          name='full_feature_low')
                # conv_concat = tf.concat([mconv2, sconv2], axis=3)
                # conv_feature_low = layers.conv2d(conv_concat, 1, 1, activation_fn=None, name='conv_feature_low')
            with tf.variable_scope('actor_low'):
                actor_hidden_low = layers.fully_connected(full_feature_low, 256, activation_fn=tf.nn.relu,
                                                           name='actor_hidden_low')
                action_low = layers.fully_connected(actor_hidden_low, 4096, activation_fn=tf.nn.softmax,
                                                  name='action_low')
            with tf.variable_scope('critic_high'):
                critic_hidden_low = layers.fully_connected(full_feature_low, 32, activation_fn=tf.nn.relu,
                                                           name='critic_hidden_low')
                value_low = tf.reshape(layers.fully_connected(critic_hidden_low, 1, activation_fn=tf.tanh, name='value_low'), [-1])

            actor_params_low = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_low')
            critic_params_low = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_low')
            feature_params_low = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='feature_low')
            # 可以在此处重新定义更新参数的方法（考虑为feature_high单独设计梯度）
            a_params_low = actor_params_low + feature_params_low
            c_params_low = critic_params_low + feature_params_low

            return action_low, value_low, a_params_low, c_params_low