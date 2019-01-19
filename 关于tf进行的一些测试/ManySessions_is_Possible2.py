import tensorflow as tf
s1 = tf.Session()
s2 = tf.Session()

c = tf.constant(1)
a = tf.get_variable(name='alpha', initializer=tf.constant(1))
# s1.run(tf.global_variables_initializer())

b = tf.get_variable(name='beta', initializer=tf.constant(2))
s2.run(tf.global_variables_initializer())

print(s1.run(c))
print(s2.run(c))
print(s2.run(a))
print(s1.run(a))    # 注释掉s1.run(tf.global_variables_initializer())后，这句话会报错，可见要想使用一个tf.Session()，必须保证里面的所有tf变量都已被初始化

#  值得注意的是, c作为不需要初始化的变量，可以被s1和s2都调用，
#  所以我觉得【一个会话图里包含的变量，是所有不需要初始化的变量 + 在run(tf.global_variables_initializer())之前创建的需要初始化的变量】

#  于是我觉得，应该尽量就开启唯一的一个tf.Session()，不然应该会很麻烦。。。