import time
import threading
import random
import numpy as np
import math
from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.lib import features
import tensorflow as tf
import tensorflow.contrib.layers as layers

import sys
from absl import flags

flags.DEFINE_string('f', '', 'kernel')
flags.FLAGS(sys.argv)

tf.set_random_seed(1)
config = tf.ConfigProto(allow_soft_placement=True)  # auto distribute device
config.gpu_options.allow_growth = True  # gpu memory dependent on require

# ———————————————— 定义工具函数 —————————————— #


def action_micro(marco_id, micro_sort_id):
    train_scv = [1, 2, 490, 3, 5]
    build_supply = [1, 3, 5, 91, 3, 5]
    build_barrack = [1, 3, 5, 42, 3, 5]
    train_marine = [1, 2, 477, 477, 477, 477, 477, 3, 5]
    idle_worker = [6, 269]
    all_army_attack = [7, 13]
    list_actions = {0: train_scv, 1: build_supply, 2: build_barrack, 3: train_marine, 4: idle_worker,
                    5: all_army_attack}

    call_low = False
    real_action_id = list_actions[marco_id][micro_sort_id]

    # ———————————— 部分宏动作已定好，无需调下层即可得动作 ———————————— #

    if marco_id == 0 and micro_sort_id == 0:  # (0,0) 移动镜头到标准位置
        real_action_id = 1
        action_args = [[20, 25]]
    elif marco_id == 0 and micro_sort_id == 1:  # (0,1) 选中基地
        real_action_id = 2
        action_args = [[0], [30, 24]]
    elif marco_id == 0 and micro_sort_id == 3:  # (0,3) 框中一些农民
        real_action_id = 3
        action_args = [[0], [5, 5], [30, 25]]
    elif marco_id == 1 and micro_sort_id == 0:  # (1,0) 移动镜头到标准位置
        real_action_id = 1
        action_args = [[20, 25]]
    elif marco_id == 1 and micro_sort_id == 1:  # (1,1) 框中一些农民
        real_action_id = 3
        action_args = [[0], [5, 5], [30, 25]]
    elif marco_id == 1 and micro_sort_id == 4:  # (1,4) 框中一些农民
        real_action_id = 3
        action_args = [[0], [5, 5], [30, 25]]
    elif marco_id == 2 and micro_sort_id == 0:  # (2,0) 移动镜头到标准位置
        real_action_id = 1
        action_args = [[20, 25]]
    elif marco_id == 2 and micro_sort_id == 1:  # (2,1) 框中一些农民
        real_action_id = 3
        action_args = [[0], [5, 5], [30, 25]]
    elif marco_id == 2 and micro_sort_id == 4:  # (2,4) 框中一些农民
        real_action_id = 3
        action_args = [[0], [5, 5], [30, 25]]
    elif marco_id == 3 and micro_sort_id == 0:  # (3,0) 移动镜头到标准位置
        real_action_id = 1
        action_args = [[20, 25]]
    elif marco_id == 3 and micro_sort_id == 7:  # (3,7) 框中一些scv
        real_action_id = 3
        action_args = [[0], [5, 5], [30, 25]]

    # ——————— 未确定宏动作需转化宏id和微id为实际id，参数call下层网络 ———————— #

    else:
        action_args = []
        for arg in actions.FUNCTIONS[real_action_id].args:
            if arg.name in ('screen', 'minimap', 'screen2'):
                call_low = True
            else:
                action_args.append([0])

        call_low = True

    return call_low, real_action_id, action_args


def building_counter(state, next_state, supply_num, barrack_num):
    # 建造得分计算，补给站是100，兵营是150，指挥中心是400
    score_change = next_state.observation["score_cumulative"][4] - state.observation["score_cumulative"][4]
    
    if score_change == 100:  # 补给站数目
        supply_num += 1
    if score_change == -100:
        supply_num -= 1
    
    if score_change == 150:  # 兵营数目
        barrack_num += 1
    if score_change == -150:
        barrack_num -= 1
        
    return supply_num, barrack_num


def if_last(marco_id, micro_sort_id):
    train_scv = [1, 2, 490, 3, 5]
    build_supply = [1, 3, 5, 91, 3, 5]
    build_barrack = [1, 3, 5, 42, 3, 5]
    train_marine = [1, 2, 477, 477, 477, 477, 477, 3, 5]
    idle_worker = [6, 269]
    all_army_attack = [7, 13]
    list_actions = {0: train_scv, 1: build_supply, 2: build_barrack, 3: train_marine, 4: idle_worker,
                    5: all_army_attack}
    if micro_sort_id < len(list_actions[marco_id]) - 1:
        return False
    else:
        return True


def high_reward(state, next_state, real_action_id, step, supply_num, barrack_num):
    reward = 0

    minerals = next_state.observation.player.minerals
    last_minerals = state.observation.player.minerals

    minerals_change = minerals - last_minerals  # 矿改变量
    idle_worker = next_state.observation.player.idle_worker_count  # 空闲工人

    army_change = next_state.observation["player"][5] - state.observation["player"][5]  # 军队变化
    worker_change = next_state.observation["player"][6] - state.observation["player"][6]  # 农民变化
    food_remain = next_state.observation["player"][4] - next_state.observation["player"][3]  # 剩余人口

    if real_action_id is 140 or 143 or 144 or 168:  # 惩罚取消类动作
        reward -= 10
    if not len(state.observation.last_actions) and real_action_id != 1:  # 惩罚无效操作
        reward -= 10

    if minerals >= 500 and minerals_change >= 0:  # 矿变化相关reward
        reward -= 50
    if 500 > minerals >= 200 and minerals_change >= 0:
        reward += 20 - 0.2 * (minerals - 200)
    if 0 <= minerals < 200 and minerals_change > 0:
        reward += minerals / 10

    if idle_worker > 0:  # 闲置农民惩罚
        reward -= 20 * idle_worker

    if 0 < food_remain < 2:  # 剩余人口奖惩
        reward -= 10 * food_remain
    if food_remain <= 0:
        reward -= 500

    worker_count = next_state.observation["player"][6]  # 农民数量奖惩
    if worker_change > 0 and worker_count <= 22:
        reward += 10
    if worker_change > 0 and worker_count > 22:
        reward -= 400

    army_count = next_state.observation["player"][5]  # 军队数量奖惩
    if army_count > 0:
        reward += 30 * army_count
    if step >= 500 and army_count == 0:
        reward -= 300
    if army_change > 0:
        reward += 50

    score_change = next_state.observation["score_cumulative"][4] - state.observation["score_cumulative"][4]

    if step >= 300 and supply_num == 0:  # 建造奖惩
        reward -= 500
    if step >= 500 and barrack_num == 0:
        reward -= 400
    if step >= 50 and state.observation.player.food_workers <= 12:
        reward -= 100
    if score_change > 0:
        reward += 3 * score_change

    kill_units_change = 10 * (next_state.observation["score_cumulative"][5]
                              - state.observation["score_cumulative"][5])
    kill_structures_change = 10 * (next_state.observation["score_cumulative"][6]
                                   - state.observation["score_cumulative"][6])

    if kill_units_change > 0:  # 击杀单元及建筑奖励
        reward += kill_units_change
    if kill_structures_change > 0:
        reward += kill_structures_change

    if step > 1000:  # 存活18分钟，人口大于12，基地没被打爆即胜利
        reward += ((step - 1000) / 10)
    if step >= 3000 and state.observation.player.food_workers >= 12 \
            and state.observation.score_cumulative.total_value_structures >= 400:
        reward += 500

    reward = float(reward / 1000)  # 归一化到正负1之间
    if reward >= 1.0:
        reward = 1.0
    if reward <= -1.0:
        reward = -1.0

    return reward


def low_reward(state, next_state, location, macro_action_id):
    reward = 0

    base = [20, 25]  # minimap
    enemy = [44, 39]  # minimap
    enemy_2 = [20, 39]  # minimap
    defense = [40, 25]  # minimap
    defense_base = [25, 25]  # minimap
    barrack = [15, 35]  # screen
    supply = [40, 25]  # screen
    build_score_change = next_state.observation["score_cumulative"][4] - state.observation["score_cumulative"][4]
    killed_score_units_change = 10 * (
            next_state.observation["score_cumulative"][5] - state.observation["score_cumulative"][5])
    killed_score_structures_change = 10 * (
            next_state.observation["score_cumulative"][6] - state.observation["score_cumulative"][6])
    army_change = next_state.observation["player"][5] - state.observation["player"][5]  # 军队变化

    if not len(state.observation.last_actions):  # 执行位置有误
        reward -= 100

    if macro_action_id == 1:  # 建造补给站的距离
        distance = math.sqrt((location[0] - supply[0]) ** 2 + (location[1] - supply[1]) ** 2)
        if 1 < distance <= 5:
            reward += 100 - distance * 20
        elif distance <= 1:
            reward = 0
        else:
            reward -= distance * 100

        if build_score_change == 100:
            reward += 300

        if reward > 1000:
            reward = 1000
        if reward < -1000:
            reward = -1000
        reward = float(reward / 1000)
        return reward

    if macro_action_id == 2:  # 建造兵营的距离
        distance = math.sqrt((location[0] - barrack[0]) ** 2 + (location[1] - barrack[1]) ** 2)
        if 2 < distance <= 5:
            reward += 100 - distance * 20
        elif distance <= 2:
            reward = 0
        else:
            reward += 100 - distance * 10

        if build_score_change == 150:
            reward += 500

        if reward > 1000:
            reward = 1000
        if reward < -1000:
            reward = -1000
        reward = float(reward / 1000)
        return reward

    if macro_action_id == 5:  # 对敌方操作，坐标类型为minimap

        dis_atk = math.sqrt((location[0] - enemy[0]) ** 2 + (location[1] - enemy[1]) ** 2)
        dis_atk_2 = math.sqrt((location[0] - enemy_2[0]) ** 2 + (location[1] - enemy_2[1]) ** 2)
        dis_def = math.sqrt((location[0] - defense[0]) ** 2 + (location[1] - defense[1]) ** 2)
        dis_def_base = math.sqrt((location[0] - defense_base[0]) ** 2 + (location[1] - defense_base[1]) ** 2)
        distance = min(dis_atk, dis_atk_2, dis_def, dis_def_base)
        reward += 200 - distance * 5

        if killed_score_units_change > 0:
            reward += 10 * killed_score_units_change
        if killed_score_structures_change > 0:
            reward += 10 * killed_score_structures_change

        if reward > 1000:
            reward = 1000
        if reward < -1000:
            reward = -1000
        reward = float(reward / 1000)

    else:  # 对己方操作，坐标类型为screen
        distance = math.sqrt((location[0] - base[0]) ** 2 + (location[1] - base[1]) ** 2)
        if 2 < distance <= 20:
            reward += 200 - distance * 2
        elif distance <= 2:
            reward = 0
        else:
            reward += 100 - distance * 10

        if build_score_change > 0:
            if build_score_change == 150:
                reward += 100
            elif build_score_change == 100:
                reward += 50

        if army_change > 0:
            reward += 500

        if killed_score_units_change > 0:
            reward += killed_score_units_change
        if killed_score_structures_change > 0:
            reward += killed_score_structures_change

        if reward > 1000:
            reward = 1000
        if reward < -1000:
            reward = -1000
        reward = float(reward / 1000)

    return reward


# ———————————————— 定义网络 ———————————————— #


def preprocess_minimap(feature_minimap):  # TODO 两个preprocess函数待解析
    minimap = np.array(feature_minimap, dtype=np.float32)
    layers = []
    for i in range(len(features.MINIMAP_FEATURES)):
        if i == features.MINIMAP_FEATURES.player_id.index:
            layers.append(minimap[i:i + 1] / features.MINIMAP_FEATURES[i].scale)
        elif features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:
            layers.append(minimap[i:i + 1] / features.MINIMAP_FEATURES[i].scale)
        else:
            layer = np.zeros([features.MINIMAP_FEATURES[i].scale, minimap.shape[1], minimap.shape[2]], dtype=np.float32)
            for j in range(features.MINIMAP_FEATURES[i].scale):
                indy, indx = (minimap[i] == j).nonzero()
                layer[j, indy, indx] = 1
            layers.append(layer)
    return np.expand_dims(np.concatenate(layers, axis=0), axis=0)


def preprocess_screen(feature_screen):
    screen = np.array(feature_screen, dtype=np.float32)
    layers = []
    for i in range(len(features.SCREEN_FEATURES)):
        if i == features.SCREEN_FEATURES.player_id.index or i == features.SCREEN_FEATURES.unit_type.index:
            layers.append(screen[i:i + 1] / features.SCREEN_FEATURES[i].scale)
        elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
            layers.append(screen[i:i + 1] / features.SCREEN_FEATURES[i].scale)
        else:
            layer = np.zeros([features.SCREEN_FEATURES[i].scale, screen.shape[1], screen.shape[2]], dtype=np.float32)
            for j in range(features.SCREEN_FEATURES[i].scale):
                indy, indx = (screen[i] == j).nonzero()
                layer[j, indy, indx] = 1
            layers.append(layer)
    return np.expand_dims(np.concatenate(layers, axis=0), axis=0)


class A3C:  # TODO 注意：传参名称为macro-micro，网络占位符名称为high-low
    def __init__(self, sess, reuse):
        self.sess = sess
        self.action_num = len(actions.FUNCTIONS)

        self.minimap = tf.placeholder(tf.float32, [None, 17, 64, 64])
        self.screen = tf.placeholder(tf.float32, [None, 42, 64, 64])

        self.info = tf.placeholder(tf.float32, [None, self.action_num])

        self.low_choose_need = tf.placeholder(tf.float32, [None])
        self.low_choose_mask = tf.placeholder(tf.float32, [None, 64 ** 2])
        self.high_choose_mask = tf.placeholder(tf.float32, [None, 6])

        self.low_q_target_value = tf.placeholder(tf.float32, [None])
        self.high_q_target_value = tf.placeholder(tf.float32, [None])

        with tf.variable_scope('agent') and tf.device('/gpu:0'):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            # ———————————————— 特征提取网络 —————————————————— #

            # TODO 上下层info不一致
            mconv1 = layers.conv2d(tf.transpose(self.minimap, [0, 2, 3, 1]), 16, 5, scope='mconv1')
            mconv2 = layers.conv2d(mconv1, 32, 3, scope='mconv2')
            sconv1 = layers.conv2d(tf.transpose(self.screen, [0, 2, 3, 1]), 16, 5, scope='sconv1')
            sconv2 = layers.conv2d(sconv1, 32, 3, scope='sconv2')
            info_feature = layers.fully_connected(layers.flatten(self.info), 256, activation_fn=tf.tanh,
                                                  scope='info_feature')

            flatten_concat = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2), info_feature], axis=1)
            flatten_feature = layers.fully_connected(flatten_concat, 256, activation_fn=tf.nn.relu,
                                                     scope='flatten_feature')

            conv_concat = tf.concat([mconv2, sconv2], axis=3)
            conv_feature = layers.conv2d(conv_concat, 1, 1, activation_fn=None, scope='conv_feature')
            # conv_tanh = layers.conv2d(conv_concat, 1, 1, activation_fn=tf.tanh, scope='conv_tanh')

            # ———————————————— Actor宏动作层 —————————————————— #

            self.high_action_output = layers.fully_connected(flatten_feature, 6, activation_fn=tf.nn.softmax,
                                                             scope='high_action_output')

            # ———————————————— Critic宏评价层 —————————————————— #

            self.high_q_value = tf.reshape(layers.fully_connected(flatten_feature, 1, activation_fn=None,
                                                                  scope='high_q_value'), [-1])

            # ———————————————— Actor底动作层 —————————————————— #

            self.low_action_output = tf.nn.softmax(layers.flatten(conv_feature))

            # ———————————————— Critic底评价层 —————————————————— #

            self.low_q_value = tf.reshape(layers.fully_connected(flatten_feature, 1, activation_fn=None,
                                                                 scope='low_q_value'), [-1])

            # ———————————————— 底层loss计算 —————————————————— # TODO Critic的优化还需细读A3C论文

            low_advantage = tf.stop_gradient(self.low_q_target_value - self.low_q_value)
            low_value_loss = 0.25 * tf.reduce_mean(tf.square(self.low_q_value * low_advantage))

            low_choose_prob = tf.reduce_sum(self.low_action_output * self.low_choose_mask, axis=1)
            low_log_prob = tf.log(tf.clip_by_value(low_choose_prob, 1e-10, 1.))

            low_action_log_prob = self.low_choose_need * low_log_prob  # TODO 是否需要传入该参数
            low_policy_loss = - tf.reduce_mean(low_action_log_prob * low_advantage)

            # ———————————————— 上层loss计算 —————————————————— #

            high_advantage = tf.stop_gradient(self.high_q_target_value - self.high_q_value)
            high_value_loss = 0.25 * tf.reduce_mean(tf.square(self.high_q_value * high_advantage))

            high_choose_prob = tf.reduce_sum(self.high_action_output * self.high_choose_mask, axis=1)
            high_log_prob = tf.log(tf.clip_by_value(high_choose_prob, 1e-10, 1.))

            high_policy_loss = - tf.reduce_mean(high_log_prob * high_advantage)

            # ———————————————— 合并loss —————————————————— #

            low_loss = low_policy_loss + low_value_loss
            high_loss = high_policy_loss + high_value_loss

            # ———————————————— 训练定义 —————————————————— #

            low_train_function = tf.train.RMSPropOptimizer(5e-4, decay=0.99, epsilon=1e-10)
            low_gradient = low_train_function.compute_gradients(low_loss)
            low_cliped_gradient = []
            for gradient, variable in low_gradient:
                if gradient is not None:
                    gradient = tf.clip_by_norm(gradient, 10.0)
                low_cliped_gradient.append([gradient, variable])
            self.low_train_op = low_train_function.apply_gradients(low_cliped_gradient)

            high_train_function = tf.train.RMSPropOptimizer(5e-4, decay=0.99, epsilon=1e-10)
            high_gradient = high_train_function.compute_gradients(high_loss)
            high_cliped_gradient = []
            for gradient, variable in high_gradient:
                if gradient is not None:
                    gradient = tf.clip_by_norm(gradient, 10.0)
                high_cliped_gradient.append([gradient, variable])
            self.high_train_op = high_train_function.apply_gradients(high_cliped_gradient)

    def choose_high_action(self, state):
        minimap = preprocess_minimap(state.observation['feature_minimap'])
        screen = preprocess_screen(state.observation['feature_screen'])

        info = np.zeros([1, self.action_num], dtype=np.float32)
        info[0, state.observation['available_actions']] = 1  # 将可执行动作置1

        feed = {self.minimap: minimap, self.screen: screen, self.info: info}

        macro_action_output = self.sess.run(self.high_action_output, feed_dict=feed)

        # TODO 筛选high_action(state.observation['available_actions'])

        macro_action_id = np.argmax(macro_action_output)

        if np.random.rand() < 0.2:
            macro_action_id = np.random.randint(0, 5)

        return macro_action_id

    def choose_low_action(self, state):
        minimap = preprocess_minimap(state.observation['feature_minimap'])
        screen = preprocess_screen(state.observation['feature_screen'])

        info = np.zeros([1, self.action_num], dtype=np.float32)
        info[0, state.observation['available_actions']] = 1

        feed = {self.minimap: minimap, self.screen: screen, self.info: info}

        micro_spatial_action = self.sess.run([self.low_action_output], feed_dict=feed)

        location = np.argmax(micro_spatial_action)
        location = [int(location // 64), int(location % 64)]
        if np.random.rand() < 0.2:  # TODO 随机选择坐标?
            noise = np.random.randint(-4, 5)
            location[0] = int(max(0, min(64 - 1, location[0] + noise)))
            location[1] = int(max(0, min(64 - 1, location[1] + noise)))

        return location[0], location[1]

    def update_low(self, buffer):
        last_state = buffer[-1][-1]

        if last_state.last():
            running_reward = 0
        else:
            minimap = preprocess_minimap(last_state.observation['feature_minimap'])
            screen = preprocess_screen(last_state.observation['feature_screen'])
            info = np.zeros([1, self.action_num], dtype=np.float32)
            info[0, last_state.observation['available_actions']] = 1

            feed = {self.minimap: minimap,
                    self.screen: screen,
                    self.info: info}

            running_reward = self.sess.run(self.low_q_value, feed_dict=feed)[0]

        minimaps = []
        screens = []
        infos = []
        rewards = []

        low_choose_need = np.zeros([len(buffer)], dtype=np.float32)
        low_choose_mask = np.zeros([len(buffer), 64 ** 2], dtype=np.float32)

        for i, [state, action_id, action_args, reward, _] in enumerate(buffer):
            minimap = preprocess_minimap(state.observation['feature_minimap'])
            screen = preprocess_screen(state.observation['feature_screen'])
            info = np.zeros([1, self.action_num], dtype=np.float32)
            info[0, state.observation['available_actions']] = 1

            minimaps.append(minimap)
            screens.append(screen)
            infos.append(info)
            rewards.append(reward)

            args = actions.FUNCTIONS[action_id].args  # TODO 这部分具体意义待测试
            for arg, act_arg in zip(args, action_args):
                if arg.name in ('screen', 'minimap', 'screen2'):
                    ind = act_arg[1] * 64 + act_arg[0]
                    low_choose_need[i] = 1
                    low_choose_mask[i, ind] = 1

        minimaps = np.concatenate(minimaps, axis=0)
        screens = np.concatenate(screens, axis=0)
        infos = np.concatenate(infos, axis=0)

        micro_q_target_value = np.zeros([len(buffer)], dtype=np.float32)
        running_add = running_reward
        for j in reversed(range(0, len(rewards))):
            running_add = rewards[j] + 0.99 * running_add
            micro_q_target_value[j] = running_add

        feed = {self.minimap: minimaps,
                self.screen: screens,
                self.info: infos,
                self.low_q_target_value: micro_q_target_value,
                self.low_choose_need: low_choose_need,
                self.low_choose_mask: low_choose_mask}
        self.sess.run(self.low_train_op, feed_dict=feed)

    def update_high(self, buffer):
        last_state = buffer[-1][-1]

        if last_state.last():
            running_reward = 0
        else:
            minimap = preprocess_minimap(last_state.observation['feature_minimap'])
            screen = preprocess_screen(last_state.observation['feature_screen'])
            info = np.zeros([1, self.action_num], dtype=np.float32)
            info[0, last_state.observation['available_actions']] = 1

            feed = {self.minimap: minimap,
                    self.screen: screen,
                    self.info: info}

            running_reward = self.sess.run(self.high_q_value, feed_dict=feed)[0]

        minimaps = []
        screens = []
        infos = []
        rewards = []

        macro_choose_mask = np.zeros([len(buffer), 6], dtype=np.float32)

        for i, [state, macro_action_id, reward, _] in enumerate(buffer):
            minimap = preprocess_minimap(state.observation['feature_minimap'])
            screen = preprocess_screen(state.observation['feature_screen'])
            info = np.zeros([1, self.action_num], dtype=np.float32)
            info[0, state.observation['available_actions']] = 1

            minimaps.append(minimap)
            screens.append(screen)
            infos.append(info)
            rewards.append(reward)

            macro_choose_mask[i, macro_action_id] = 1

        minimaps = np.concatenate(minimaps, axis=0)
        screens = np.concatenate(screens, axis=0)
        infos = np.concatenate(infos, axis=0)

        macro_q_target_value = np.zeros([len(buffer)], dtype=np.float32)
        running_add = running_reward
        for j in reversed(range(0, len(rewards))):
            running_add = rewards[j] + 0.99 * running_add
            macro_q_target_value[j] = running_add

        feed = {self.minimap: minimaps,
                self.screen: screens,
                self.info: infos,
                self.high_q_target_value: macro_q_target_value,
                self.high_choose_mask: macro_choose_mask}
        self.sess.run(self.high_train_op, feed_dict=feed)


# ———————————————— 定义执行过程 —————————————— #


def build_env(map_name):
    if map_name == 'Simple64':
        players = [sc2_env.Agent(sc2_env.Race['terran']),
                   sc2_env.Bot(sc2_env.Race['terran'], sc2_env.Difficulty['very_easy'])]
    else:
        players = [sc2_env.Agent(sc2_env.Race['terran'])]
    interface = sc2_env.parse_agent_interface_format(feature_screen=64, feature_minimap=64)
    env = sc2_env.SC2Env(map_name=map_name, players=players, step_mul=8, agent_interface_format=interface)
    return env


def run(agent, max_epoch, map_name):
    env = build_env(map_name)
    micro_buffer = []
    macro_buffer = []

    for epoch in range(max_epoch):
        state = env.reset()[0]
        step = 0
        micro_running = False
        macro_action_id, micro_action_id, supply_num, barrack_num = 0, 0, 0, 0
        location = [0, 0]  # 避免警告，实际无用
        while True:  # 每一个step执行一个动作
            step += 1
            if not micro_running:  # 微动作执行完毕，或无微动作执行
                macro_action_id = agent.choose_high_action(state)
                micro_running = True

            call_low, real_action_id, action_args = action_micro(macro_action_id, micro_action_id)
            if if_last(macro_action_id, micro_action_id):
                micro_action_id = 0
                micro_running = False
            else:
                micro_action_id += 1

            if call_low:
                location = [0, 0]
                location[0], location[1] = agent.choose_low_action(state)
                action_args = []
                for arg in actions.FUNCTIONS[real_action_id].args:
                    if arg.name in ('screen', 'minimap', 'screen2'):
                        action_args.append(location)
                    else:
                        action_args.append([0])  # TODO 未知该参数意义

            if real_action_id not in state.observation["available_actions"]:
                real_action_id = 0
                action_args = []

            print("动作选择", real_action_id, action_args)
            next_state = env.step([actions.FunctionCall(real_action_id, action_args)])[0]

            supply_num, barrack_num = building_counter(state, next_state, supply_num, barrack_num)

            if call_low:
                micro_reward = low_reward(state, next_state, location, macro_action_id)
                micro_buffer.append([state, real_action_id, action_args, micro_reward, next_state])
            else:
                macro_reward = high_reward(state, next_state, real_action_id, step, supply_num, barrack_num)
                macro_buffer.append([state, macro_action_id, macro_reward, next_state])

            if len(micro_buffer) > 50:  # TODO buffer学习的判断具体设置（num_frames >= max_frames）
                agent.update_low(micro_buffer)  # TODO 是否加入lr衰减
                micro_buffer = []

            if len(macro_buffer) > 10:  # TODO buffer学习的判断具体设置（num_frames >= max_frames）
                agent.update_high(macro_buffer)  # TODO 是否加入lr衰减
                macro_buffer = []

            if next_state.last():
                if micro_buffer is not None and macro_buffer is not None:
                    agent.update_low(micro_buffer)
                    agent.update_high(macro_buffer)
                    micro_buffer = []
                    macro_buffer = []
                break

            state = next_state

    env.close()


def run_a3c(max_epoch, map_name, parallel):
    sess = tf.Session(config=config)

    agents = []
    for i in range(parallel):
        agent = A3C(sess, i > 0)
        agents.append(agent)

    sess.run(tf.global_variables_initializer())

    threads = []
    for i in range(parallel):
        t = threading.Thread(target=run, args=(agents[i], max_epoch, map_name))
        threads.append(t)
        t.daemon = True
        t.start()
        time.sleep(5)

    coord = tf.train.Coordinator()
    coord.join(threads)


# ———————————————— 主程序 ———————————————— #

max_epoch = 40000
map_name = 'Simple64'
# parallel = 4  # GeForce GTX1080Ti
parallel = 2  # GeForce GTX1070

run_a3c(max_epoch, map_name, parallel)
