import time
import threading
import numpy as np
from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.lib import features
from pysc2 import maps
from pysc2.lib import stopwatch
import tensorflow as tf
import tensorflow.contrib.layers as layers
import globalvar as GL
from a3c_reward import a3c_reward
from macro_actions import action_micro

list_actions, _ = GL.get_list()

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


class A3C:
    def __init__(self, sess, reuse):
        # self.action_num = len(actions.FUNCTIONS)
        self.action_num = 6  # non_spatial输出宏动作id，spatial输出location

        # 每个agent单独的会话、单独的观测传参
        self.sess = sess
        self.minimap = tf.placeholder(tf.float32, [None, 17, 64, 64])
        self.screen = tf.placeholder(tf.float32, [None, 42, 64, 64])
        # self.info = tf.placeholder(tf.float32, [None, self.action_num])
        # 为了适配宏动作，把所有的info（available_actions）都去除了

        self.spatial_mask = tf.placeholder(tf.float32, [None])
        self.spatial_choose = tf.placeholder(tf.float32, [None, 64 ** 2])
        self.non_spatial_mask = tf.placeholder(tf.float32, [None, self.action_num])
        self.non_spatial_choose = tf.placeholder(tf.float32, [None, self.action_num])
        self.q_target_value = tf.placeholder(tf.float32, [None])

        with tf.variable_scope('agent') and tf.device('/gpu:0'):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            # ———————————————— 特征提取网络 —————————————————— #

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

    def step(self, state, env, thread_index):
        minimap = preprocess_minimap(state.observation['feature_minimap'])
        screen = preprocess_screen(state.observation['feature_screen'])
        # info = np.zeros([1, self.action_num], dtype=np.float32)
        # info[0, state.observation['available_actions']] = 1  
        feed = {self.minimap: minimap, self.screen: screen}  # self.info: info
        non_spatial_action, spatial_action = self.sess.run(
            [self.non_spatial_action, self.spatial_action],
            feed_dict=feed)

        non_spatial_action = non_spatial_action.ravel()  # TODO ravel函数作用待解释
        spatial_action = spatial_action.ravel()
        valid_actions = [0, 1, 2, 3, 4, 5]
        # action_id = valid_actions[np.argmax(non_spatial_action[valid_actions])]    # TODO：valid_action设置是否正确
        macro_id = np.argmax(non_spatial_action)
        if np.random.rand() < 0.05:  # 随机选择动作 epsilon-greedy
            macro_id = np.random.choice(valid_actions)

        location = np.argmax(spatial_action)
        location = [int(location // 64), int(location % 64)]  # 注意坐标的x,y顺序是反的
        if np.random.rand() < 0.2:  # epsilon-greedy
            noise = np.random.randint(-4, 5)
            location[0] = int(max(0, min(64 - 1, location[0] + noise)))
            location[1] = int(max(0, min(64 - 1, location[1] + noise)))
        location.reverse()  # 后面再用的时候，顺序就对了，注意

        action_args = []
        ind_last = GL.get_value(thread_index, "ind_micro")
        if ind_last == -1 or ind_last == -99 or ind_last == 666:
            # ind_last == -99 (表示宏动作里的微动作执行失败)
            # ind_last == 666 (表示宏动作成功执行完毕）
            GL.set_value(thread_index, "dir_high", macro_id)
            GL.set_value(thread_index, "ind_micro", 0)
            ind_todo = GL.get_value(thread_index, "ind_micro")
        else:
            temp = GL.get_value(thread_index, "ind_micro")
            GL.set_value(thread_index, "ind_micro", temp + 1)
            ind_todo = GL.get_value(thread_index, "ind_micro")

        macro_id = GL.get_value(thread_index, "dir_high")  # non_spatial_action
        action, call_spatial, action_id, macro_type, coord_type = action_micro(thread_index, macro_id,
                                                                               ind_todo)
        # 关键一步，调用了macro_action.action_micro计算出选择的action和其他参数。
        # 如果call_spatial为False，则act_id没用了，直接使用上行中的action
        # 如果其为True，则进入以下的模块，action没用了，act_id被使用来计算新的action
        if call_spatial:
            GL.set_value(thread_index, "act_id_micro", ind_todo)
            if macro_id == 2 and ind_todo == 3:  # 代表当前要执行的动作是宏动作“build_barrack”中的序号3微动作：造兵营（动作函数42）
                GL.set_value(thread_index, "barrack_location_NotSure", [location[0], location[1]])
            # action_args = []
            for arg in actions.FUNCTIONS[action_id].args:  # actions是pysc2.lib中的文件 根据act_id获取其可使用的参数，并添加到args中去
                if arg.name in ('screen', 'minimap', 'screen2'):
                    action_args.append([location[0], location[1]])
                else:
                    action_args.append([0])  # TODO: Be careful
                action = [actions.FunctionCall(action_id, action_args)]

        # 校验宏动作：
        flag_success = True
        if list_actions[macro_id][ind_todo] not in state.observation['available_actions']:
            GL.set_value(thread_index, "ind_micro", -99)  # 表示宏动作里的微动作执行失败
            action = [actions.FunctionCall(function=0, arguments=[])]  # 执行no_op
            flag_success = False
            # 当ind_todo是最后一个需要执行的动作，且执行成功时，将ind_done[ind_thread]设为666（即宏动作成功执行完毕）
        if ind_todo == len(list_actions[macro_id]) - 1 and flag_success:
            GL.set_value(thread_index, "ind_micro", 666)  # 表示宏动作执行到了最后一步微动作且执行成功
        next_state = env.step(action)[0]  # env环境的step函数根据动作计算出下一个step
        # 动作函数合法但失败（比如造补给站在available_action_list里，但选的建造坐标在基地的位置上，则造不出来），则将ind_micro置为-99，表示“宏动作执行失败”
        if not len(next_state.observation.last_actions) and action[0].function != 1:
            GL.set_value(thread_index, "ind_micro", -99)
        # 代表当前要执行的动作是宏动作“build_barrack”中的序号3微动作：造兵营（动作函数42）,【且执行成功（ind_micro != 99）】（也有可能没成功，比如造了一半农民走了or造了一半被敌方拆了）
        if macro_id == 2 and ind_todo == 3 and GL.get_value(thread_index, "ind_micro") != -99:
            barrack_location = GL.get_value(thread_index, "barrack_location_NotSure")
            GL.add_value_list(thread_index, "barrack_location", barrack_location)

        ######
        # ——————— 加入epsilon随机值衰减 ——————— #
        # epsilon = [0.05, 0.2]
        # if counter >= 2000:
        #     epsilon[0] = epsilon[0] / 1.005
        #     epsilon[1] = epsilon[1] / 1.005
        #
        # if np.random.rand() < epsilon[0]:
        #     action_id = np.random.choice(valid_actions)
        # if np.random.rand() < epsilon[1]:
        #     noise = np.random.randint(-4, 5)
        #     location[0] = int(max(0, min(64 - 1, location[0] + noise)))
        #     location[1] = int(max(0, min(64 - 1, location[1] + noise)))        
        # valid_actions = state.observation['available_actions']

        return action_id, action_args, next_state, flag_success, location, macro_id, macro_type, coord_type

    def update(self, buffer, epoch, thread_index):
        last_state = buffer[-1][-1]
        if last_state.last():
            R = 0  # TODO R设置的具体意义？
        else:
            minimap = preprocess_minimap(last_state.observation['feature_minimap'])
            screen = preprocess_screen(last_state.observation['feature_screen'])
            # info = np.zeros([1, self.action_num], dtype=np.float32)
            # info[0, last_state.observation['available_actions']] = 1
            feed = {self.minimap: minimap, self.screen: screen}  # self.info: info
            R = self.sess.run(self.q_value, feed_dict=feed)[0]

        minimaps = []
        screens = []
        # infos = []
        rewards = []
        # ————— 原计算target value的设计 ————————— #
        # target_value = np.zeros([len(buffer)], dtype=np.float32)
        # target_value[-1] = R

        spatial_mask = np.zeros([len(buffer)], dtype=np.float32)
        spatial_choose = np.zeros([len(buffer), 64 ** 2], dtype=np.float32)
        non_spatial_mask = np.zeros([len(buffer), self.action_num], dtype=np.float32)
        non_spatial_choose = np.zeros([len(buffer), self.action_num], dtype=np.float32)

        for i, [state, macro_id, action_id, action_args, reward, next_state] in enumerate(buffer):
            # 求解每个step的minimap、screen、info（已去除）、reward
            minimap = preprocess_minimap(state.observation['feature_minimap'])
            screen = preprocess_screen(state.observation['feature_screen'])
            # info = np.zeros([1, self.action_num], dtype=np.float32)
            # info[0, state.observation['available_actions']] = 1

            # 求解mask和choose，non-spatial:宏动作号，spatial：坐标
            # valid_actions = state.observation["available_actions"]
            valid_actions = [0, 1, 2, 3, 4, 5]
            non_spatial_mask[i, valid_actions] = 1  # 可用命令列表
            non_spatial_choose[i, macro_id] = 1

            args = actions.FUNCTIONS[action_id].args  # TODO 这部分具体意义待测试
            for arg, act_arg in zip(args, action_args):
                # act_arg就是spatial的坐标
                if arg.name in ('screen', 'minimap', 'screen2'):
                    ind = act_arg[0] * 64 + act_arg[1]  # 注意坐标的顺序
                    spatial_mask[i] = 1  # 是否需要spatial
                    spatial_choose[i, ind] = 1

            minimaps.append(minimap)
            screens.append(screen)
            # infos.append(info)
            rewards.append(reward)
            # ——————— 原计算target value的设计 ————————— #
            # target_value[i] = reward + 0.99 * target_value[i - 1]

        minimaps = np.concatenate(minimaps, axis=0)
        screens = np.concatenate(screens, axis=0)
        # infos = np.concatenate(infos, axis=0)

        # 求解q_target_value
        q_target_value = np.zeros([len(buffer)], dtype=np.float32)
        running_add = R
        for j in reversed(range(0, len(rewards))):
            running_add = rewards[j] + 0.99 * running_add
            q_target_value[j] = running_add

        feed = {self.minimap: minimaps,
                self.screen: screens,
                # self.info: infos,
                self.q_target_value: q_target_value,
                self.spatial_mask: spatial_mask,
                self.spatial_choose: spatial_choose,
                self.non_spatial_mask: non_spatial_mask,
                self.non_spatial_choose: non_spatial_choose}
        self.sess.run(self.train_op, feed_dict=feed)

        if sum(rewards) > 2:  # TODO: 资格迹，优先经验重放
            print("good_episode: ", epoch)

    def save_model(self, path, count):
        # GL.set_saving(True)
        self.saver.save(self.sess, path + '/model.ckpt', count)

    def load_model(self, path):
        ckpt = tf.train.get_checkpoint_state(path)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        return int(ckpt.model_checkpoint_path.split('-')[-1])


def build_env(map_name):
    if map_name == 'Simple64':
        players = [sc2_env.Agent(sc2_env.Race['terran']),
                   sc2_env.Bot(sc2_env.Race['terran'], sc2_env.Difficulty['very_easy'])]
    else:
        players = [sc2_env.Agent(sc2_env.Race['terran'])]
    interface = sc2_env.parse_agent_interface_format(feature_screen=64, feature_minimap=64)
    env = sc2_env.SC2Env(map_name=map_name, players=players, step_mul=8, agent_interface_format=interface)
    return env


def run(agent, max_epoch, map_name, thread_index, flags, snapshot_path):
    env = build_env(map_name)
    buffer = []
    thread_index_all = flags.parallel
    max_step = flags.max_agent_steps

    for episode in range(max_epoch):
        # episode
        episode = GL.get_episode_counter()
        state = env.reset()[0]  # ==timesteps[0]
        counter = 0  # step_counter
        GL.episode_init(thread_index)
        while True:
            # step
            counter += 1
            GL.set_value(thread_index, "num_steps", counter)
            action_id, action_args, next_state, flag_success, location, macro_id, macro_type, coord_type = agent.step(
                state, env,
                thread_index)
            reward = a3c_reward(thread_index, next_state, state, flag_success, location, macro_id, macro_type,
                                coord_type)
            sum_reward = GL.get_value(thread_index, "sum_reward")
            sum_reward += reward
            GL.set_value(thread_index, "sum_reward", sum_reward)
            GL.add_value_list(thread_index, "reward_of_episode", reward)

            buffer.append([state, macro_id, action_id, action_args, reward, next_state])
            if counter % 200 == 1:    # max_step是10的倍数，不能和此处的同余，否则就会把清空的buffer传进去update
                agent.update(buffer, episode, thread_index)  # TODO 是否加入lr衰减
                buffer = []

            if counter > max_step or next_state.last():  # 最终状态
                if buffer is not None:
                    agent.update(buffer, episode, thread_index)  # TODO 是否加入lr衰减
                buffer = []
                state = next_state  # 获取终末state
                with threading.Lock():  # 线程锁，为了读写安全
                    episode = GL.get_episode_counter()
                    episode += 1
                    GL.set_episode_counter(episode)
                break  # 结束本episode的运行，继续执行后续的存储语句
            state = next_state

        # 存储episode数据
        episode_log(state, episode, thread_index, counter, thread_index_all, flags, snapshot_path, agent)

    env.close()


def episode_log(state, episode, thread_index, num_step, thread_index_all, flags, snapshot_path, agent):
    iswin = state.reward
    score = state.observation["score_cumulative"][0]
    print("Episode_counter: ", episode)
    print("state.reward_isWin: ", iswin)
    print('Episode score:  ', score)
    GL.add_value_list(thread_index, "victory_or_defeat", iswin)
    episode_reward_average = GL.get_value(thread_index, "sum_reward") / num_step
    GL.add_value_list(thread_index, "reward_list", episode_reward_average)
    GL.add_value_list(thread_index, "episode_score_list", score)
    # 存储全episode的累积数据
    GL.add_value_list(thread_index_all, "victory_or_defeat", iswin)
    GL.add_value_list(thread_index_all, "episode_score_list", score)
    # global_episode是FLAGS.snapshot_step的倍数+1，或指定回合数
    # 存单个episode的reward变化，存储网络参数（tf.train.Saver().save(),见a3c_agent），存全局numpy以备急停
    if (episode % flags.snapshot_step == 1) or (episode in flags.quicksave_step_list):
        agent.save_model(snapshot_path, episode)
        for i in range(flags.parallel):
            np.save(
                "./DataForAnalysis/reward_of_episode_" + str(episode) + "thread_" + str(i) + ".npy",
                GL.get_value(i, "reward_of_episode"))
            np.save(
                "./DataForAnalysis/reward_list_thread_" + str(i) + "episode_" + str(episode) + ".npy",
                GL.get_value(i, "reward_list"))
            np.save("./DataForAnalysis/victory_or_defeat_thread_" + str(i) + "episode_" + str(
                episode) + ".npy",
                    GL.get_value(i, "victory_or_defeat"))
            np.save("./DataForAnalysis/episode_score_list_thread_" + str(i) + "episode_" + str(
                episode) + ".npy",
                    GL.get_value(i, "episode_score_list"))
        # 存储全episode的累积数据
        np.save("./DataForAnalysis/victory_or_defeat_thread" + str(thread_index_all) + "episode" + str(
            episode) + ".npy", GL.get_value(thread_index_all, "victory_or_defeat"))
        np.save("./DataForAnalysis/episode_score_list_thread" + str(thread_index_all) + "episode" + str(
            episode) + ".npy", GL.get_value(thread_index_all, "episode_score_list"))


def run_a3c(max_epoch, map_name, parallel, flags, snapshot_path):
    sess = tf.Session(config=config)
    stopwatch.sw.enabled = flags.profile or flags.trace  # 应该是开启类似计时时钟这样的观测量
    stopwatch.sw.trace = flags.trace
    maps.get(flags.map)  # Assert the map exists.

    agents = []
    for i in range(parallel):
        agent = A3C(sess, i > 0)
        agents.append(agent)

    sess.run(tf.global_variables_initializer())
    # 模型读取
    if not flags.training or flags.continuation:  # 若不是训练模式 或 若是持续性训练，则利用原有数据（训练好的参数，存在了snapshot文件夹里）进行训练
        COUNTER = agent.load_model(snapshot_path)
        GL.set_episode_counter(COUNTER)
        print("Parameter loaded, global_episode_counter = ", COUNTER)
        # 全局变量COUNTER记录的是当前所有线程加在一起，总共完成的回合数

    threads = []
    for i in range(parallel):
        t = threading.Thread(target=run, args=(agents[i], max_epoch, map_name, i, flags, snapshot_path))
        threads.append(t)
        t.daemon = True
        t.start()
        time.sleep(5)

    coord = tf.train.Coordinator()
    coord.join(threads)
