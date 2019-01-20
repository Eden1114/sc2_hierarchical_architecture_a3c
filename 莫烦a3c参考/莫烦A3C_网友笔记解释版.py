# DHN 总结 start：
# 本篇代码原地址：https://blog.csdn.net/Tiberium_discover/article/details/84198209
# 这篇代码的注释有一些错误的地方，我自行做了修改，比如计算a_loss的过程中，ta对一些变量内容的解释不对（比如acts—prob）

# 可以看出，该程序应当取自莫烦源代码中的A3C_discrete_action.py，但这个程序和我一直以来参考的A3C_continuous_action.py中有些部分不太一样

# 比如很重要的a_loss的计算，就和continuous中的写法不太一样，但比continuous中的写法简单。考虑到星际的环境也是离散型输出（宏动作编号是离散的数字，spatial_action也是离散的坐标点）
# 所以我觉得在设计星际的a_loss时可以参考这个A3C_discrete_action.py中设计的方法

# c_loss的设计在A3C_discrete_action.py和A3C_continuous_action.py中都是一样的

# 另外这份代码，跟我下载的莫烦的A3C_discrete_action.py虽然差不多一样，但有一点小区别，
# 比如没有ENTROPY_BETA = 0.001（探索几率）和相应的增加探索度的操作

# DHN 总结 end

import multiprocessing  #多线程模块
import threading  #线程模块
import tensorflow as tf
import numpy as np
import gym
import os
import shutil  #拷贝文件用
import matplotlib.pyplot as plt

Game='CartPole-v0'
N_workers=multiprocessing.cpu_count()    #独立玩家个体数为cpu数
MAX_GLOBAL_EP=2000  #中央大脑最大回合数
GLOBALE_NET_SCOPE='Globale_Net' #中央大脑的名字
UPDATE_GLOBALE_ITER=10   #中央大脑每N次提升一次
GAMMA=0.9    #衰减度
LR_A=0.0001   #Actor网络学习率
LR_C=0.001    #Critic 网络学习率

GLOBALE_RUNNING_R=[]   #存储总的reward
GLOBALE_EP=0   #中央大脑步数

env=gym.make(Game)   #定义游戏环境


N_S=env.observation_space.shape[0]  #观测值个数
N_A=env.action_space.n              #行为值个数


class ACnet(object):     #这个class即可用于生产global net，也可生成 worker net，因为结构相同
    def __init__(self,scope,globalAC=None):   #scope 用于确定生成什么网络
        if scope==GLOBALE_NET_SCOPE:   #创建中央大脑
            with tf.variable_scope(scope):
                self.s=tf.placeholder(tf.float32,[None,N_S],'S')   #初始化state，None代表batch，N—S是每个state的观测值个数
                self.build_net(scope)                               #建立中央大脑神经网络
                self.a_params=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope+'/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                #定义中央大脑actor和critic的参数


        else:                        #创建worker两个网络的具体步骤
            with tf.variable_scope(scope):    #这里的scope传入的是worker的名字
                self.s=tf.placeholder(tf.float32,[None,N_S],'S')  #初始化state
                self.a_his = tf.placeholder(tf.int32, [None,1], 'A_his')         #初始化action,是一个[batch，1]的矩阵，第二个维度为1，
                                                                                  #格式类似于[[1],[2],[3]]
                self.v_target=tf.placeholder(tf.float32,[None,1],'Vtarget')     #初始化v现实(V_target)，数据格式和上面相同


                self.acts_prob,self.v=self.build_net(scope)   #建立神经网络，acts_prob为返回的概率值,v为返回的评价值
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')


                td=tf.subtract(self.v_target,self.v,name='TD_error')  #计算td—error即v现实和v估计之差
                                                                      #v—target和v都是一串值，v-target（现实）已经计算好并传入了，v估计由传入的
                                                                      #一系列state送入critic网络确定

                with tf.name_scope('c_loss'):    #计算Critic网络的loss
                    self.c_loss=tf.reduce_mean(tf.square(td))  #Critic的loss就是td—error加平方避免负数


                with tf.name_scope('a_loss'):    #计算actor网络的损失
                    log_prob = tf.reduce_sum(tf.log(self.acts_prob +1e-5)*tf.one_hot(self.a_his,N_A,dtype=tf.float32),axis=1,keep_dims=True)

                    # 网友（非DHN）的解释：
                    #这里是矩阵乘法，目的是筛选出本batch曾进行的一系列选择的概率值，acts—prob类似于一个向量[0.3,0.8,0.5]，
                    #one—hot是在本次进行的的操作置位1，其他位置置为0，比如走了三次a—his为[1,0,3],N—A是4，则one—hot就是[[0,1,0,0],[1,0,0,0],[0,0,0,1]]
                    #相乘以后就是[[0,0.3,0,0],[0.8,0,0,0],[0,0,0,0.5]],log_prob就是计算这一系列选择的log值。

                    # 我（DHN）对其进行更正后：
                    # N_A = 2，代表行为值个数。即只有2个数（0，1  0表示左，1表示右，应该）
                    # 假设更新时经历的step数为3
                    # 这里是矩阵乘法，目的是筛选出本batch曾进行的一系列选择的概率值，acts—prob维度应该是“更新时经历的step数”* “N_A(2)”  形如[[0.2, 0.8],[0.4, 0.6],[0.3, 0.7]]
                    # one—hot维度同上，是在本次进行的的操作置位1，其他位置置为0。  a—his维度应该是“更新时经历的step数”，形如[1,0,1](N—A是2)， 则one—hot就是[[0,1],[1,0],[1,0]]
                    # 相乘以后就是[[0, 0.8],[0.4, 0],[0.3, 0]], 再reduce_sum就变成[0.8, 0.4, 0.3]。

                    self.exp_v = log_prob * td     #td决定梯度下降的方向（但在我下载的莫烦a3c里这里要stop_gradient）
                    #  ！！！！！！！！！！ exp_v、log_prob、td维度均为“更新时经历的step数”

                    self.a_loss=tf.reduce_mean(-self.exp_v)    #计算actor网络的损失a-loss


                with tf.name_scope('local_grad'):
                    self.a_grads=tf.gradients(self.a_loss,self.a_params)   #实现a_loss对a_params每一个参数的求导，返回一个list
                    self.c_grads=tf.gradients(self.c_loss,self.c_params)   #实现c_loss对c_params每一个参数的求导，返回一个list


            with tf.name_scope('sync'):   #worker和global的同步过程
                with tf.name_scope('pull'):   #获取global参数,复制到local—net
                    self.pull_a_params_op=[l_p.assign(g_p) for l_p,g_p in zip(self.a_params,globalAC.a_params)]
                    self.pull_c_params_op=[l_p.assign(g_p) for l_p,g_p in zip(self.c_params,globalAC.c_params)]
                with tf.name_scope('push'):   #将参数传送到gloabl中去
                    self.update_a_op=OPT_A.apply_gradients(zip(self.a_grads,globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))
                    #其中传送的是local—net的actor和critic的参数梯度grads,具体计算在上面定义
                    #apply_gradients是tf.train.Optimizer中自带的功能函数，将求得的梯度参数更新到global中


    def build_net(self,scope): #建立神经网络过程
        w_init=tf.random_normal_initializer(0.,.1)  #初始化神经网络weights
        with tf.variable_scope('actor'):            #actor神经网络结构
            l_a=tf.layers.dense(inputs=self.s,units=200,activation=tf.nn.relu6,
                                kernel_initializer=w_init,bias_initializer=tf.constant_initializer(0.1),name='la')  #建立第一层神经网络
            acts_prob=tf.layers.dense(inputs=l_a,units=N_A,activation=tf.nn.softmax,
                               kernel_initializer=w_init,bias_initializer=tf.constant_initializer(0.1),name='act_prob')  #第二层神经网络其中之一输出为动作的均值

        with tf.variable_scope('critic'):     #critic神经网络结构,输入为位置的观测值，输出为评价值v
            l_c=tf.layers.dense(self.s,20,tf.nn.relu6,kernel_initializer=w_init,bias_initializer=tf.constant_initializer(0.1),name='lc')  #建立第一层神经网络
            v=tf.layers.dense(l_c,1,kernel_initializer=w_init,bias_initializer=tf.constant_initializer(0.1),name='v')   #第二层神经网络

        return acts_prob,v    #建立神经网络后返回的是输入当前state得到的actor网络的动作概率和critic网络的v估计


    def update_global(self,feed_dict):    #定义更新global参数函数
        SESS.run([self.update_a_op,self.update_c_op],feed_dict)    #分别更新actor和critic网络

    def pull_global(self):   #定义更新local参数函数
        SESS.run([self.pull_a_params_op,self.pull_c_params_op])

    def choose_action(self,s):   #定义选择动作函数
        s=s[np.newaxis, :]
        probs=SESS.run(self.acts_prob, feed_dict={self.s: s})
        return np.random.choice(range(probs.shape[1]), p=probs.ravel())   #从probs中按概率选取出某一个动作



class Worker(object):
    def __init__(self,name,globalAC):    #传入的name是worker的名字，globalAC是已经建立好的中央大脑GLOBALE—AC
        self.env=gym.make(Game).unwrapped
        self.name=name                   #worker的名字
        self.AC=ACnet(name,globalAC)     #第二个参数当传入的是已经建立好的GLOBALE—AC时创建的是local net
                                         #建立worker的AC网络

    def work(self):   #定义worker运行的的具体过程
        global  GLOBALE_RUNNING_R,GLOBALE_EP   #两个全局变量，R是所有worker的总reward，ep是所有worker的总episode
        total_step=1                            #本worker的总步数
        buffer_s,buffer_a,buffer_r=[],[],[]    #state,action,reward的缓存

        while not COORD.should_stop() and GLOBALE_EP<MAX_GLOBAL_EP:   #停止本worker运行的条件
                                                                     #本循环一次是一个回合

            s=self.env.reset()       #初始化环境
            ep_r=0                   #本回合总的reward

            while True:      #本循环一次是一步
                if self.name=='W_0':    #只有worker0才将动画图像显示
                    self.env.render()

                a=self.AC.choose_action(s)    #将当前状态state传入AC网络选择动作action
                # print('hhhhhhhh == ' ,a)

                s_,r,done,info=self.env.step(a)   #行动并获得新的状态和回报等信息

                if done:r=-5    #如果结束了，reward给一个惩罚数

                ep_r+=r              #记录本回合总体reward
                buffer_s.append(s)   #将当前状态，行动和回报加入缓存
                buffer_a.append(a)
                buffer_r.append(r)


                if total_step % UPDATE_GLOBALE_ITER==0 or done:  #每iter步完了或者或者到达终点了，进行同步sync操作
                    if done:
                        v_s_=0   #如果结束了，设定对未来的评价值为0
                    else:
                        v_s_=SESS.run(self.AC.v,feed_dict={self.AC.s:s_[np.newaxis,:]})[0,0]   #如果是中间步骤，则用AC网络分析下一个state的v评价

                    buffer_v_target=[]
                    for r in buffer_r[::-1]:    #将下一个state的v评价进行一个反向衰减传递得到每一步的v现实
                        v_s_=r + GAMMA* v_s_
                        buffer_v_target.append(v_s_)  #将每一步的v现实都加入缓存中
                    buffer_v_target.reverse()    #反向后，得到本系列操作每一步的v现实(v-target)

                    buffer_s,buffer_a,buffer_v_target=np.vstack(buffer_s),np.vstack(buffer_a),np.vstack(buffer_v_target)

                    feed_dict={
                        self.AC.s:buffer_s,                 #本次走过的所有状态，用于计算v估计
                        self.AC.a_his:buffer_a,             #本次进行过的所有操作，用于计算a—loss
                        self.AC.v_target:buffer_v_target    #走过的每一个state的v现实值，用于计算td
                    }

                    self.AC.update_global(feed_dict)  #update—global的具体过程在AC类中定义，feed-dict如上

                    buffer_s,buffer_a,buffer_r=[],[],[]   #清空缓存

                    self.AC.pull_global()    #从global—net提取出参数赋值给local—net

                s=s_   #跳转到下一个状态
                total_step+=1  #本回合总步数加1


                if done:   #如果本回合结束了
                    if len(GLOBALE_RUNNING_R)==0:  #如果尚未记录总体running
                        GLOBALE_RUNNING_R.append(ep_r)
                    else:
                        GLOBALE_RUNNING_R.append(0.9*GLOBALE_RUNNING_R[-1]+0.1*ep_r)

                    print(self.name,'EP:',GLOBALE_EP)
                    GLOBALE_EP+=1       #加一回合
                    break   #结束本回合



if __name__=='__main__':
    SESS=tf.Session()

    with tf.device('/cpu:0'):
        OPT_A=tf.train.RMSPropOptimizer(LR_A,name='RMSPropA')    #定义actor训练过程,后续主要是使用该optimizer中的apply—gradients操作
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')  #定义critic训练过程
        GLOBALE_AC=ACnet(GLOBALE_NET_SCOPE)  #创建中央大脑GLOBALE_AC，只创建结构(A和C的参数)
        workers=[]
        for i in range(N_workers):    #N—workers等于cpu数量
            i_name='W_%i'%i   #worker name
            workers.append(Worker(name=i_name,globalAC=GLOBALE_AC))   #创建独立的worker

        COORD=tf.train.Coordinator()    #多线程
        SESS.run(tf.global_variables_initializer())   #初始化所有参数

        worker_threads=[]
        for worker in workers:    #并行过程
            job= lambda:worker.work()   #worker的工作目标,此处调用Worker类中的work
            t=threading.Thread(target=job)  #每一个线程完成一个worker的工作目标
            t.start()                 # 启动每一个worker
            worker_threads.append(t)   #每一个worker的工作都加入thread中
        COORD.join(worker_threads)     #合并几个worker,当每一个worker都运行完再继续后面步骤

        plt.plot(np.arange(len(GLOBALE_RUNNING_R)),GLOBALE_RUNNING_R)   #绘制reward图像
        plt.xlabel('step')
        plt.ylabel('Total moving reward')
        plt.show()
