import tensorflow as tf
import numpy as np


x = tf.constant([[0,1.,0], [0,1.,0]])
y = tf.constant([[0.1, 0.6, 0.3], [0.2, 0.3, 0.5]])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print( sess.run(tf.reduce_sum(x * y, axis=1)) )
    # print( sess.run(tf.log(tf.constant([2.73, 1]))))