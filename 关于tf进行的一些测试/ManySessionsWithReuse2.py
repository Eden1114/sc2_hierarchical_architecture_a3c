import tensorflow as tf

def getV():
    with tf.variable_scope('Big', reuse=tf.AUTO_REUSE):
        m = tf.get_variable(name='man', initializer=tf.random_uniform([1]))
    return m

a = getV()
s1 = tf.Session()
s1.run(tf.global_variables_initializer())

b = getV()
s2 = tf.Session()
s2.run(tf.global_variables_initializer())

print(a)
print(b)
print(s1.run(a))
print(s2.run(b))    # 这个程序就是Reuse2的缩小版，更简明地展示了Reuse2为什么和Reuse3不一样（变量不共享），
                    # 因为【同名的变量“Big/man:0”其实是在2个tf会话图！】，所以即使同名，它们的值、内存位置都不一样
                    # 就像桌面上有2个文件夹，2个文件里都有叫作“Big/man:0”的文件，但这两个“Big/man:0”的值和内存位置就不一样
