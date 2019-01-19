import tensorflow as tf

a = tf.get_variable(name='alpha', initializer=tf.constant(1))
s1 = tf.Session()
s1.run(tf.global_variables_initializer())   # 这句话的位置对于“一个会话图里包含哪些tf节点”是决定性的！ 而s1 = tf.Session()只要在这句话之前就行，但是不是在创建tf节点的语句之前创建就无所谓！


b = tf.get_variable(name='beta', initializer=tf.constant(2))
s2 = tf.Session()
s2.run(tf.global_variables_initializer())


print(s2.run(a))    # 这句话不会报错，证明了s2的图结构包含了s2.run(tf.global_variables_initializer())之前定义过的所有tf节点，而不是包含了s2 = tf.Session()之前定义过的所有tf节点

print(s1.run(b))    # 这句话会报错，证明了s1的图结构只包含了s1.run(tf.global_variables_initializer()) 之前定义过的所有tf节点，而不是s1 = tf.Session()之前定义过的所有tf节点
                    # 但这句话报错同时证明了【可以建立 多个 tf会话图，不止一个】，因为显然s1和s2都可以被使用，但二者并不一样