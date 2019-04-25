from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers

w_init = tf.random_normal_initializer(0., .1)


def build_high_net(minimap, screen, info, num_macro_action):
    with tf.variable_scope('network_high'):
        with tf.variable_scope('feature_high'):
            mconv1 = layers.conv2d(tf.transpose(minimap, [0, 2, 3, 1]), 16, 5, scope='mconv1')
            mpool1 = tf.nn.max_pool(mconv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            mconv2 = layers.conv2d(mpool1, 32, 3, scope='mconv2')
            mpool2 = tf.nn.max_pool(mconv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            sconv1 = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]), 16, 5, scope='sconv1')
            spool1 = tf.nn.max_pool(sconv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            sconv2 = layers.conv2d(spool1, 32, 3, scope='sconv2')
            spool2 = tf.nn.max_pool(sconv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            info_high = layers.fully_connected(layers.flatten(info), 16, activation_fn=None,
                                               scope='info_high')
            info_feat = layers.fully_connected(info_high, 128, activation_fn=tf.nn.relu,
                                               scope='info_feat')

            full_concat = tf.concat([layers.flatten(mpool2), layers.flatten(spool2), info_feat], axis=1)
            # print(full_concat.shape)
            # full_concat.shape = 262176[conv=32*5]；131088[conv=16*5]；16400[max_pool]；16512[pool+info_feat]
            # full_feature_high = layers.fully_connected(full_concat, 256, activation_fn=tf.nn.relu,
            #                                            scope='full_feature_high')
        with tf.variable_scope('actor_high'):
            actor_hidden_high = layers.fully_connected(full_concat, 64, activation_fn=tf.nn.relu,
                                                       scope='actor_hidden_high_1')
            actor_hidden_high_2 = layers.fully_connected(actor_hidden_high, 16, activation_fn=tf.nn.relu,
                                                         scope='actor_hidden_high_2')
            action_high_prob = tf.layers.dense(actor_hidden_high_2, num_macro_action, activation=tf.nn.softmax,
                                               kernel_initializer=w_init, name='action_high_prob')
        with tf.variable_scope('critic_high'):
            critic_hidden_high = layers.fully_connected(full_concat, 64, activation_fn=tf.nn.relu,
                                                        scope='critic_hidden_high')
            critic_hidden_high_2 = layers.fully_connected(critic_hidden_high, 16, activation_fn=tf.nn.relu,
                                                        scope='critic_hidden_high_2')
            value_high = tf.reshape(
                layers.fully_connected(critic_hidden_high_2, 1, activation_fn=tf.tanh, scope='value_high'), [-1])

        actor_params_high = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='network_high/actor_high')
        critic_params_high = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='network_high/critic_high')
        feature_params_high = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='network_high/feature_high')
        # 可以在此处重新定义更新参数的方法（考虑为feature_high单独设计梯度）
        a_params_high = actor_params_high + feature_params_high
        c_params_high = critic_params_high + feature_params_high

        return action_high_prob, value_high, a_params_high, c_params_high


def build_low_net(minimap, screen, info, spatial_size):
    with tf.variable_scope('network_low'):
        with tf.variable_scope('feature_low'):
            # full_feature_low = layers.fully_connected(full_concat, 256, activation_fn=tf.nn.relu,
            #                                           scope='full_feature_low')
            mconv1 = layers.conv2d(tf.transpose(minimap, [0, 2, 3, 1]), 16, 5, scope='mconv1')
            mpool1 = tf.nn.max_pool(mconv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            mconv2 = layers.conv2d(mpool1, 32, 3, scope='mconv2')
            mpool2 = tf.nn.max_pool(mconv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            sconv1 = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]), 16, 5, scope='sconv1')
            spool1 = tf.nn.max_pool(sconv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            sconv2 = layers.conv2d(spool1, 32, 3, scope='sconv2')
            spool2 = tf.nn.max_pool(sconv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            info_low = layers.fully_connected(layers.flatten(info), 16, activation_fn=None,
                                               scope='info_high')
            info_feat = layers.fully_connected(info_low, 128, activation_fn=tf.nn.relu,
                                               scope='info_feat')

            full_concat = tf.concat([layers.flatten(mpool2), layers.flatten(spool2), info_feat], axis=1)
        with tf.variable_scope('actor_low'):
            actor_hidden_low = layers.fully_connected(full_concat, 128, activation_fn=tf.nn.relu,
                                                      scope='actor_hidden_low_1')
            actor_hidden_low_2 = layers.fully_connected(actor_hidden_low, 512, activation_fn=tf.nn.relu,
                                                        scope='actor_hidden_low_2')
            action_low_prob = tf.layers.dense(actor_hidden_low_2, spatial_size, activation=tf.nn.softmax,
                                              kernel_initializer=w_init,
                                              name='action_low_prob')
        with tf.variable_scope('critic_low'):
            critic_hidden_low = layers.fully_connected(full_concat, 64, activation_fn=tf.nn.relu,
                                                       scope='critic_hidden_low')
            critic_hidden_low_2 = layers.fully_connected(critic_hidden_low, 16, activation_fn=tf.nn.relu,
                                                       scope='critic_hidden_low_2')
            value_low = tf.reshape(
                layers.fully_connected(critic_hidden_low_2, 1, activation_fn=tf.tanh, scope='value_low'), [-1])

        actor_params_low = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='network_low/actor_low')
        critic_params_low = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='network_low/critic_low')
        feature_params_low = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='network_low/feature_low')
        # 可以在此处重新定义更新参数的方法（考虑为feature_high单独设计梯度）
        a_params_low = actor_params_low + feature_params_low
        c_params_low = critic_params_low + feature_params_low

        return action_low_prob, value_low, a_params_low, c_params_low
