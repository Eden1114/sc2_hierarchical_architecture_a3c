import tensorflow as tf

a = tf.Variable(2.0)
b = tf.Variable(3.0)

c = tf.add(a, b)

c_stoped = tf.stop_gradient(c)

d = tf.add(a, b)

e_1= tf.add(c, d)
e_2 = tf.add(c_stoped, d)

gradients_1 = tf.gradients(e_1, xs=[a, b])
gradients_2 = tf.gradients(e_2, xs=[a, b])
gradients_3 = tf.gradients(c_stoped, xs=[a, b])

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(gradients_1))    # 输出为2 2 因为e_1 = a+b + a+b = 2a + 2b
    print(sess.run(gradients_2))    # 输出为1 1 因为e = "a+b" + a+b  加引号的部分被阻挡了对于梯度的计算
    print(gradients_3)    # 输出为None 道理同上
