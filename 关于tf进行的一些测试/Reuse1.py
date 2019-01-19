import tensorflow as tf

def createVar():
    # 如果没有reuse参数，y = createVar()会报错，因为不能存在2个名字同为V/alpha:0的变量
    # 如果reuse参数为True，x = createVar()会报错，因为名字为V/alpha:0的变量还并不存在，无法复用
    # 如果reuse参数为tf.AUTO_REUSE，程序才能运行正确，因为x = createVar()会新生成变量，y = createVar()会复用该变量
    # ——结果就是，x和y在tf图结构中，【名字(V/alpha:0)和值和 所 在 内 存 空 间 都一样】，设计的初衷是为了节约变量存储空间
    with tf.variable_scope('V', reuse=tf.AUTO_REUSE):
        a = tf.get_variable(name='alpha', shape=[1],
                            initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0, dtype=tf.float32, seed=None))
    return a


with tf.Session() as sess:
    x = createVar()
    y = createVar()
    print(x)    # 可见,虽然没有经历tf.global_variables_initializer()和sess.run()，tf张量中除了数值以外的信息其实是available的
    print(y)

    sess.run(tf.global_variables_initializer())
    print(sess.run(x))
    print(sess.run(y), end='\n\n')

    print(sess.run(tf.assign_add(x, tf.ones(1))))
    print(sess.run(y))  # sess.run()直观地来看有一种套壳儿的感觉，以前的变量直接拿来用就可以，tf中的张量则要sess.run()后才可使用，使用前要套这么一层壳子的感觉。
    # print(sess.run(tf.add(x, tf.ones(1))))
    # print(sess.run(y))


# tensorflow的add的两种写法。
# a = tf.assign_add(a, tf.ones(1))     tf.assign_add(a, tf.ones(1))会返回一个【和a内存位置相同】的变量（但返回值值在tensor图结构中的名字变成了一个新的名字，不同于本来a的名字）
# a = tf.add(a, tf.ones(1))            tf.add(a, tf.ones(1))会返回一个新的【和a内存位置不同】的变量（且返回值值在tensor图结构中的名字也是一个新的名字，不同于本来a的名字）
# 可以参考博文
# https://blog.csdn.net/silent56_th/article/details/75563344