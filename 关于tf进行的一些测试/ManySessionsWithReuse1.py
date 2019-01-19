import tensorflow as tf

def getV():
    with tf.variable_scope('Big', reuse=tf.AUTO_REUSE):
        m = tf.get_variable(name='man', initializer=tf.random_uniform([1]))
    return m

a = getV()
b = getV()
s1 = tf.Session()
s1.run(tf.global_variables_initializer())

print(a)
print(b)
print(s1.run(a))
print(s1.run(a))    # 可见不同的python变量a，b指向【同一个tf会话图里的唯一的一个“Big/man:0”】
