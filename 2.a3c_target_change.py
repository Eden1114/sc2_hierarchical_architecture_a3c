import sys
import time
import threading
import numpy as np
from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.lib import features
import tensorflow as tf
import tensorflow.contrib.layers as layers
from absl import flags
flags.DEFINE_string('f', '', 'kernel')  # jupyter运行问题
flags.FLAGS(sys.argv)

tf.set_random_seed(1)
config = tf.ConfigProto(allow_soft_placement=True)  # auto distribute device
config.gpu_options.allow_growth = True  # gpu memory dependent on require


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


class Agent:
    def __init__(self, sess):
        # 每个agent单独的会话、单独的观测传参
        self.sess = sess
        self.minimap = tf.placeholder(tf.float32, [None, 17, 64, 64])
        self.screen = tf.placeholder(tf.float32, [None, 42, 64, 64])
        self.info = tf.placeholder(tf.float32, [None, action_num])

        self.spatial_mask = tf.placeholder(tf.float32, [None])
        self.spatial_choose = tf.placeholder(tf.float32, [None, 64 ** 2])
        self.non_spatial_mask = tf.placeholder(tf.float32, [None, action_num])
        self.non_spatial_choose = tf.placeholder(tf.float32, [None, action_num])
        self.q_target_value = tf.placeholder(tf.float32, [None])

    def build(self, reuse):
        with tf.variable_scope('agent') and tf.device('/gpu:0'):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            # ———————————————— 特征提取网络 —————————————————— #
            
            mconv1 = layers.conv2d(tf.transpose(self.minimap, [0, 2, 3, 1]), 16, 5, scope='mconv1')
            mconv2 = layers.conv2d(mconv1, 32, 3, scope='mconv2')
            sconv1 = layers.conv2d(tf.transpose(self.screen, [0, 2, 3, 1]), 16, 5, scope='sconv1')
            sconv2 = layers.conv2d(sconv1, 32, 3, scope='sconv2')
            info_feature = layers.fully_connected(layers.flatten(self.info), 256, activation_fn=tf.tanh,
                                                  scope='info_feature')

            flatten_concat = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2), info_feature], axis=1)
            flatten_feature = layers.fully_connected(flatten_concat, 256, activation_fn=tf.nn.relu,
                                                    scope='flatten_feature')

            # TODO 可能加入info向量信息到conv层
            conv_concat = tf.concat([mconv2, sconv2], axis=3)
            conv_feature = layers.conv2d(conv_concat, 1, 1, activation_fn=None, scope='conv_feature')

            # ———————————————— 动作选择输出网络 —————————————————— #

            self.q_value = tf.reshape(layers.fully_connected(flatten_feature, 1, activation_fn=None,
                                                           scope='q_value'), [-1])  # TODO 作用未知

            self.spatial_action = tf.nn.softmax(layers.flatten(conv_feature))

            self.non_spatial_action = layers.fully_connected(flatten_feature, action_num, activation_fn=tf.nn.softmax,
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

            loss = policy_loss + value_loss

            # ———————————————— 训练定义 —————————————————— #

            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.99, epsilon=1e-10)
            grads = opt.compute_gradients(loss)
            cliped_grad = []
            for grad, var in grads:
                grad = tf.clip_by_norm(grad, 10.0)
                cliped_grad.append([grad, var])
            self.train_op = opt.apply_gradients(cliped_grad)

    def step(self, state):
        minimap = preprocess_minimap(state.observation['feature_minimap'])
        screen = preprocess_screen(state.observation['feature_screen'])

        info = np.zeros([1, action_num], dtype=np.float32)
        info[0, state.observation['available_actions']] = 1  # TODO 未知设置

        feed = {self.minimap: minimap, self.screen: screen, self.info: info}

        non_spatial_action, spatial_action = self.sess.run(
            [self.non_spatial_action, self.spatial_action],
            feed_dict=feed)

        non_spatial_action = non_spatial_action.ravel()  # TODO ravel函数作用待解释
        spatial_action = spatial_action.ravel()

        valid_actions = state.observation['available_actions']
        
        act_id = valid_actions[np.argmax(non_spatial_action[valid_actions])]
        if np.random.rand() < 0.05:  # 随机选择动作
            act_id = np.random.choice(valid_actions)

        location = np.argmax(spatial_action)
        location = [int(location // 64), int(location % 64)]
        if np.random.rand() < 0.2:  # TODO 随机选择坐标?
            noise = np.random.randint(-4, 5)
            location[0] = int(max(0, min(64 - 1, location[0] + noise)))
            location[1] = int(max(0, min(64 - 1, location[1] + noise)))

        act_args = []
        for arg in actions.FUNCTIONS[act_id].args:
            if arg.name in ('screen', 'minimap', 'screen2'):
                act_args.append([location[1], location[0]])
            else:
                act_args.append([0])  # TODO 未知该参数意义
        return actions.FunctionCall(act_id, act_args)

    def update(self, buffer):
        last_state = buffer[-1][-1]

        if last_state.last():
            R = 0  # TODO R设置的具体意义？
        else:
            minimap = preprocess_minimap(last_state.observation['feature_minimap'])
            screen = preprocess_screen(last_state.observation['feature_screen'])
            info = np.zeros([1, action_num], dtype=np.float32)
            info[0, last_state.observation['available_actions']] = 1

            feed = {self.minimap: minimap,
                    self.screen: screen,
                    self.info: info}
            
            R = self.sess.run(self.q_value, feed_dict=feed)[0]

        minimaps = []
        screens = []
        infos = []
        rewards = []

        spatial_mask = np.zeros([len(buffer)], dtype=np.float32)
        spatial_choose = np.zeros([len(buffer), 64 ** 2], dtype=np.float32)
        non_spatial_mask = np.zeros([len(buffer), action_num], dtype=np.float32)
        non_spatial_choose = np.zeros([len(buffer), action_num], dtype=np.float32)

        for i, [state, action, _] in enumerate(buffer):
            # 求解minimap、screen、info、reward
            minimap = preprocess_minimap(state.observation['feature_minimap'])
            screen = preprocess_screen(state.observation['feature_screen'])
            info = np.zeros([1, action_num], dtype=np.float32)
            info[0, state.observation['available_actions']] = 1

            reward = state.reward

            minimaps.append(minimap)
            screens.append(screen)
            infos.append(info)
            rewards.append(reward)

            # 求解mask和choose
            act_id = action.function
            act_args = action.arguments

            valid_actions = state.observation["available_actions"]
            non_spatial_mask[i, valid_actions] = 1
            non_spatial_choose[i, act_id] = 1

            args = actions.FUNCTIONS[act_id].args  # TODO 这部分具体意义待测试
            for arg, act_arg in zip(args, act_args):
                if arg.name in ('screen', 'minimap', 'screen2'):
                    ind = act_arg[1] * 64 + act_arg[0]
                    spatial_mask[i] = 1
                    spatial_choose[i, ind] = 1

        minimaps = np.concatenate(minimaps, axis=0)
        screens = np.concatenate(screens, axis=0)
        infos = np.concatenate(infos, axis=0)

        # 求解q_target_value
        q_target_value = np.zeros([len(buffer)], dtype=np.float32)
        running_add = R
        for j in reversed(range(0, len(rewards))):
            running_add = rewards[j] + 0.99 * running_add
            q_target_value[j] = running_add

        feed = {self.minimap: minimaps,
                self.screen: screens,
                self.info: infos,
                self.q_target_value: q_target_value,
                self.spatial_mask: spatial_mask,
                self.spatial_choose: spatial_choose,
                self.non_spatial_mask: non_spatial_mask,
                self.non_spatial_choose: non_spatial_choose}
        self.sess.run(self.train_op, feed_dict=feed)


def run(agent, map_name):
    players = [sc2_env.Agent(sc2_env.Race['terran']),
               sc2_env.Bot(sc2_env.Race['terran'], sc2_env.Difficulty['very_easy'])]
    interface = sc2_env.parse_agent_interface_format(feature_screen=64, feature_minimap=64)
    env = sc2_env.SC2Env(map_name=map_name, players=players, step_mul=8, agent_interface_format=interface)

    buffer = []
    for _ in range(max_epoch):
        state = env.reset()[0]

        while True:
            action = agent.step(state)
            next_state = env.step([action])[0]

            buffer.append([state, action, next_state])

            if next_state.last():  # TODO buffer学习的判断具体设置（num_frames >= max_frames）
                agent.update(buffer)  # TODO 是否加入lr衰减
                buffer = []
                break
                
            state = next_state

    env.close()


max_epoch = 4000
parallel = 3
map_name = 'MoveToBeacon'
learning_rate = 5e-4
action_num = len(actions.FUNCTIONS)

sess = tf.Session(config=config)

agents = []
for i in range(parallel):
    agent = Agent(sess)
    agent.build(i > 0)
    agents.append(agent)
sess.run(tf.global_variables_initializer())

threads = []
for i in range(parallel - 1):
    t = threading.Thread(target=run, args=(agents[i], "MoveToBeacon"))
    threads.append(t)
    t.daemon = True
    t.start()
    time.sleep(5)

run(agents[-1], "MoveToBeacon")

coord = tf.train.Coordinator()
coord.join(threads)
