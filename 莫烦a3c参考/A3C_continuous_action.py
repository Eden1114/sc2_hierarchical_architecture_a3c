"""
Asynchronous Advantage Actor Critic (A3C) with continuous action space, Reinforcement Learning.

The Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.8.0
gym 0.10.5
"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt

GAME = 'Pendulum-v0'
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
MAX_EP_STEP = 200   # 一个回合内规定可以进行的最大步数
MAX_GLOBAL_EP = 2000    # 最多可以进行的回合数
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.01
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GLOBAL_RUNNING_R = []   # 用来存储回合结束后所获得最终奖励的列表
GLOBAL_EP = 0   # 进行完的回合总数（所有线程的回合数加起来）

env = gym.make(GAME)

N_S = env.observation_space.shape[0]    # 状态state的参数的数量
N_A = env.action_space.shape[0]         # 动作action的参数的数量
A_BOUND = [env.action_space.low, env.action_space.high]     # 不太明白其含义


class ACNet(object):    # actor-critic网络（both global&local）的结构和功能定义
    def __init__(self, scope, globalAC=None):   #  因为写了globalAC=None，所以若建立ACNet时只传入scope，则globalAC=None，否则globalAC为传入的参数

        # ！！！ 根据scope的值判断是创建global还是local net
        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')   # s是state 可以看到占位符每次接受的数据是不只一组的，而是很多组（参数None）
                self.a_params, self.c_params = self._build_net(scope)[-2:]  # [-2:]意思是取_build_net(scope)函数返回值（有5个）中的后两个：actor网络的参数和critic网络的参数
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')   # 可以看到占位符每次接受的数据是不只一组的，而是很多组（参数None）
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')    # v现实，是算完以后传进来的

                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)    # 接收网络返回的五个值

                td = tf.subtract(self.v_target, self.v, name='TD_error')    # 术语TD_Error：代表“v_target(v现实)减去v(v估计)的差值”（联系Q_Learning理解,跟Q现实和Q估计的的关系很类似）
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))     # 用来修改Critic网络的方向传播值：c_loss。等于各个TD_Error平方的平均数

                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * A_BOUND[1], sigma + 1e-4

                normal_dist = tf.distributions.Normal(mu, sigma)    # mu为均值，sigma为方差，以这两个参数生成一个数值列表（供选择动作choose_a时使用）

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * tf.stop_gradient(td) # ！！！！！！ 这里stop_gradient应该是因为，用a_loss更新actor网络参数时，不应该计入critic网络参数的影响
                                                            # 而td是“update时算出的v_target”减去 “critic 网络算出的v”

                    entropy = normal_dist.entropy()  # encourage exploration  变量entropy的意义在于增加探索度，跟Q_learning的epsilon_greedy参数的意义类似
                    self.exp_v = ENTROPY_BETA * entropy + exp_v     # exp_v 是期待获得的奖励/价值，所以希望它越大越好（所以底下用它计算loss的时候它是负的）
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), A_BOUND[0], A_BOUND[1])
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):     # 下两行操作把global net中的actor和critic网络中的参数转移到local net的相应参数空间里
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):     # 下两行操作把local net中的actor和critic网络中的参数转移到 global的相应参数空间里
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))  # OPT_A在if __name__ == "__main__"入口内建立，所以此处才能调用
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):    # 该函数会建立起actor和critic两套网络，该函数返回值是分别从两个网络里拿到的
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):    # actor网络输入state输出动作a、动作概率的平均数mu和动作概率的标准差sigma
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        with tf.variable_scope('critic'):   # critic网络同样输入state输出动作a、状态state的价值v
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')  # 关键变量：a_params、c_params存储网络参数，在_build_net函数模块中生成并返回
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic') # tf.GraphKeys.TRAINABLE_VARIABLES是可学习的变量(一般指神经网络中的参数)
        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local   PUSH
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local    PULL
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local   调用了self.A，...调用了normal_dist,,,,调用了mu和sigma,...由actor网络生成（在_build_net函数里）
        s = s[np.newaxis, :]
        return SESS.run(self.A, {self.s: s})


class Worker(object):
    def __init__(self, name, globalAC):     # name是字符串，作用就是给一个名字；globalAC是一个ACNet对象
        self.env = gym.make(GAME).unwrapped     # 构造函数：生成环境env，起名name，创建该worker的网络ACNet（local net）
        self.name = name
        self.AC = ACNet(name, globalAC)  # 建立专属于当前这个worker的唯一的local net网络

    def work(self):  # work函数这里是具体更新参数的步骤
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:    # 底下发生的是一个回合内过程
            s = self.env.reset()
            ep_r = 0
            for ep_t in range(MAX_EP_STEP):      # 底下发生的是回合内一步的过程
                if self.name == 'W_0':    # 这两行的意义是：如果是第一个worker的话，可视化其运行的过程
                    self.env.render()
                a = self.AC.choose_action(s)    # 选取动作
                s_, r, done, info = self.env.step(a)
                done = True if ep_t == MAX_EP_STEP - 1 else False   # 到达预先设置的最大回合步数时，才设置done为True，即才能结束当前的回合

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r+8)/8)    # normalize 疑问：为何要加8除8？

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net  步数每增加UPDATE_GLOBAL_ITER(10)次后更新global net & local net
                    if done:
                        v_s_ = 0   # terminal 如果到了终点，则对未来的期望（v_s_是下一个状态的价值）设为0
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]    # 没到终点则根据critic网络（self.AC.v是critic网络中的输出值）计算出未来的期望/下一个状态的价值/V现实（v_s_）
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r 这几行操作比较难懂 相关内容在莫烦A3C讲解视频的24:40开始 可结合Q_Learning部分“Gamma参数的意义理解”那张图来理解
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)    # 可以看到，是一定步数（UPDATE_GLOBAL_ITER）更新
                    buffer_s, buffer_a, buffer_r = [], [], []   # 关键：！！每次更新local net/global net时会清空buffer（即更新时需要用到的输入量）
                    self.AC.pull_global()               # 可以看到，是一定步数（UPDATE_GLOBAL_ITER）更新

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],    # 回合结束时打印最后的奖励（数值在GLOBAL_RUNNING_R列表的最后一位）
                          )
                    GLOBAL_EP += 1
                    break

if __name__ == "__main__":  # 底下的操作有：1）建立优化器 2）建立工人们workers（线程）
    SESS = tf.Session()     # 3）运行tf图&启动并join线程 4）输出log(供tensorboard使用) 5）绘制step-reward曲线

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')    # 这行及下行定义了2个学习速率(LR=learning rate)不同的优化器Optimizer,还并没有实际开始优化任何网络
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params  先建立Global Net
        workers = []
        # Create worker
        for i in range(N_WORKERS):  # 有几个虚拟cpu（即线程）即创建几个worker，为的是最大化计算效率
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()  # 关于多线程的一个调度器
    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)    # 把每一份工作安排到每一份线程当中去
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()

