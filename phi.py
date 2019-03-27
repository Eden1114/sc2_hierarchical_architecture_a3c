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


_, macro_action_num = GL.get_list()


class PHI:
    def __init__(self, sess, reuse):
        self.action_num = macro_action_num  # high输出宏动作id，low输出location
        self.sess = sess
        self.lr = 1e-4

        # 每个agent单独的观测传参
        self.minimap = tf.placeholder(tf.float32, [None, prep.minimap_channel(), 64, 64])
        self.screen = tf.placeholder(tf.float32, [None, prep.screen_channel(), 64, 64])
        # self.info = tf.placeholder(tf.float32, [None, self.action_num])
        # 为了适配宏动作，把所有的info（available_actions）都去除了

        self.spatial_mask = tf.placeholder(tf.float32, [None])
        self.spatial_choose = tf.placeholder(tf.float32, [None, 64 ** 2])
        self.non_spatial_mask = tf.placeholder(tf.float32, [None, self.action_num])
        self.non_spatial_choose = tf.placeholder(tf.float32, [None, self.action_num])
        self.q_target_value = tf.placeholder(tf.float32, [None])

        self.low_choose_need = tf.placeholder(tf.float32, [None])
        self.low_choose_mask = tf.placeholder(tf.float32, [None, 64 ** 2])
        self.high_choose_mask = tf.placeholder(tf.float32, [None, 6])

        self.low_q_target_value = tf.placeholder(tf.float32, [None])
        self.high_q_target_value = tf.placeholder(tf.float32, [None])

        with tf.variable_scope('phi') and tf.device('/gpu:0'):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            # ———————————————— 特征提取网络 —————————————————— #
            with tf.variable_scope('conv_high'):
                mconv1 = layers.conv2d(tf.transpose(self.minimap, [0, 2, 3, 1]), 16, 5, scope='mconv1')
                mconv2 = layers.conv2d(mconv1, 32, 3, scope='mconv2')
                sconv1 = layers.conv2d(tf.transpose(self.screen, [0, 2, 3, 1]), 16, 5, scope='sconv1')
                sconv2 = layers.conv2d(sconv1, 32, 3, scope='sconv2')
                # info_feature = layers.fully_connected(layers.flatten(self.info), 256, activation_fn=tf.tanh,
                #                                       scope='info_feature')

                # flatten_concat = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2), info_feature], axis=1)
                flatten_concat = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2)], axis=1)
                flatten_feature = layers.fully_connected(flatten_concat, 256, activation_fn=tf.nn.relu,
                                                         scope='flatten_feature')

                # TODO 可能加入info向量信息到conv层
                conv_concat = tf.concat([mconv2, sconv2], axis=3)
                conv_feature = layers.conv2d(conv_concat, 1, 1, activation_fn=None, scope='conv_feature')

            # ———————————————— 动作选择输出网络 —————————————————— #

            self.q_value = tf.reshape(layers.fully_connected(flatten_feature, 1, activation_fn=None,
                                                             scope='q_value'), [-1])  # TODO 作用未知

            self.spatial_action = tf.nn.softmax(layers.flatten(conv_feature))

            self.non_spatial_action = layers.fully_connected(flatten_feature, self.action_num,
                                                             activation_fn=tf.nn.softmax,
                                                             scope='non_spatial_action')

            # ———————————————— 策略提升网络 —————————————————— #
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
