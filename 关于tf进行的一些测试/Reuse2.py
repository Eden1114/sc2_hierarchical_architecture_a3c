import tensorflow as tf
import numpy as np
import threading
from absl import app

x_data = np.float32(np.random.rand(2, 100))    # np.random.rand(2, 100)返回服从“0~1”均匀分布的随机样本值，数目是2 x 100（第一维100，第二维2）
                                                # np.float32将数据转换为float32型 所以x_data维度为2 x 100
y_data = np.dot([0.100, 0.200], x_data) + 0.300   # np.dot为点乘（内积）     y_data维度为 1 x 100

class Network():

    def setup(self, ind):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def model(self):
        with tf.variable_scope('V', reuse=tf.AUTO_REUSE):   # get_variable比variable高级一些
            self.b = tf.get_variable(name='beta', initializer=tf.zeros([1]))     # b维度1x1
            self.a = tf.get_variable(name='alpha', initializer=tf.random_uniform([1, 2], -1.0, 1.0)) # a维度1x2
            self.y = tf.matmul(self.a, x_data) + self.b    # a * self.x维度1x100，b维度为1x1 重点！！！！！这里根据tf（numpy也是这样）的设计，b会【自动分别加到1x100的每一个数上】
            self.loss = tf.reduce_mean(tf.square(y_data - self.y))  # tf.square(xxx)是对xxx里的每一个元素求平方（所以xxx维度完全不变）
            self.opt = tf.train.AdamOptimizer(learning_rate=0.01)
            self.train_op = self.opt.minimize(self.loss)

    def train(self, ind):    # !!!!!!!!!注意 在类中定义的变量名和函数名不要重名，不然在调用函数时可能报错（xxx is not callable）即，你可能是想使用【函数】xxx，但python会以为你想使用【变量】xxx，却画蛇添足地加了个括号）
        self.sess.run(self.train_op)
        print('index', ind, 'train is done')

def run_thread(ind, _):
    n = Network()   # KEY !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    n.model()       # 1） 先建模型！（创建tf图中的节点/Tensor/OP）
    n.setup(ind)    # 2） 再建立会话！（sess = tf.Session()）  （这步和第一步颠倒也可以，会话图里包含什么tf变量只取决于第3步的位置）
                    # 3） 再sess.run(tf.global_variables_initializer())
                    # 4） 最后再实际sess.run(想获得的变量值 or 想进行的操作)
    s = n.sess
    for t in range(1000):
        print(str(t+1) + '  index ' + str(ind) + ' beforeTrain', s.run(n.a), s.run(n.b))
        print('   index ' + str(ind) + ' beforeTrain', n.a, n.b)
        n.train(ind)
        # print(str(t + 1) + '  index ' + str(ind) + ' after Train', s.run(n.a), s.run(n.b))
        # print('   index ' + str(ind) + ' after Train', n.a, n.b)

def _main(unused_argv):
    # Run threads
    threads = []
    for i in range(2):     # 建立2个线程并运行
        t = threading.Thread( target=run_thread, args=(i+1, 0) )     # threading是python自己的线程模块，参数1为线程运行的函数名称，参数2为该函数需要的参数
                                        # args里必须是两个及以上的参数... 即run_thread(起别的名字也可以)函数必须设置为要接收两个及以上的形参个数，不然threading.Thread这句就会报错
        threads.append(t)
        t.daemon = True     # 守护进程
        t.start()

    for t in threads:   # 这个循环必须写！ 不信的话线程无法正常运行！
        t.join()

if __name__ == "__main__":
  app.run(_main)
