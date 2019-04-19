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
            # mconv2 = layers.conv2d(mconv1, 32, 3, scope='mconv2')
            sconv1 = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]), 16, 5, scope='sconv1')
            # sconv2 = layers.conv2d(sconv1, 32, 3, scope='sconv2')
            info_high = layers.fully_connected(layers.flatten(info), 32, activation_fn=None,
                                               scope='info_high')

            full_concat = tf.concat([layers.flatten(mconv1), layers.flatten(sconv1), info_high], axis=1)
            full_feature_high = layers.fully_connected(full_concat, 128, activation_fn=tf.nn.relu,
                                                       scope='full_feature_high')
        with tf.variable_scope('actor_high'):
            actor_hidden_high = layers.fully_connected(full_feature_high, 64, activation_fn=tf.nn.relu,
                                                       scope='actor_hidden_high_1')
            actor_hidden_high_2 = layers.fully_connected(actor_hidden_high, 32, activation_fn=tf.nn.relu,
                                                         scope='actor_hidden_high_2')
            action_high_prob = tf.layers.dense(actor_hidden_high_2, num_macro_action, activation=tf.nn.softmax, kernel_initializer=w_init, name='action_high_prob')
        with tf.variable_scope('critic_high'):
            critic_hidden_high = layers.fully_connected(full_feature_high, 32, activation_fn=tf.nn.relu,
                                                        scope='critic_hidden_high')
            # value_high = tf.reshape(
            #     layers.fully_connected(critic_hidden_high, 1, activation_fn=None, scope='value_high'), [-1])
            value_high = layers.fully_connected(critic_hidden_high, 1, activation_fn=None, scope='value_high')
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


def build_low_net_cst(screen, info, dir_high, act_id):    # cst：construction，建造，只使用screen
    with tf.variable_scope('network_low_cst'):
        with tf.variable_scope('feature_low_cst'):
            sconv1 = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]), 16, 5, scope='sconv1')
            sconv2 = layers.conv2d(sconv1, 32, 3, scope='sconv2')
            high_net_output = tf.concat([dir_high, act_id], axis=1)
            info_concat = tf.concat([layers.flatten(info), high_net_output], axis=1)
            info_low = layers.fully_connected(info_concat, 32, activation_fn=None,
                                              scope='info_low')

            full_concat = tf.concat([layers.flatten(sconv2), info_low], axis=1)
            full_feature_low = layers.fully_connected(full_concat, 128, activation_fn=tf.nn.relu,
                                                      scope='full_feature_low')
        with tf.variable_scope('actor_low_cst'):
            actor_hidden_low = layers.fully_connected(full_feature_low, 256, activation_fn=tf.nn.relu,
                                                      scope='actor_hidden_low_1')
            actor_hidden_low_2 = layers.fully_connected(actor_hidden_low, 1024, activation_fn=tf.nn.relu,
                                                        scope='actor_hidden_low_2')
            action_low_prob_cst = tf.layers.dense(actor_hidden_low_2, 4096, activation=tf.nn.softmax, kernel_initializer=w_init,
                                         name='action_low_prob')
        with tf.variable_scope('critic_low_cst'):
            critic_hidden_low = layers.fully_connected(full_feature_low, 32, activation_fn=tf.nn.relu,
                                                       scope='critic_hidden_low')
            value_low_cst =layers.fully_connected(critic_hidden_low, 1, activation_fn=None, scope='value_low')
    actor_params_low = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='network_low_cst/actor_low_cst')
    critic_params_low = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='network_low_cst/critic_low_cst')
    feature_params_low = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='network_low_cst/feature_low_cst')
    # 可以在此处重新定义更新参数的方法（考虑为feature_high单独设计梯度）
    a_params_low = actor_params_low + feature_params_low
    c_params_low = critic_params_low + feature_params_low

    return action_low_prob_cst, value_low_cst, a_params_low, c_params_low


def build_low_net_atk(minimap, info, dir_high, act_id):    # atk：attack，攻击，只使用minimap
    with tf.variable_scope('network_low_atk'):
        with tf.variable_scope('feature_low_atk'):
            mconv1 = layers.conv2d(tf.transpose(minimap, [0, 2, 3, 1]), 16, 5, scope='mconv1')
            mconv2 = layers.conv2d(mconv1, 32, 3, scope='mconv2')
            high_net_output = tf.concat([dir_high, act_id], axis=1)
            info_concat = tf.concat([layers.flatten(info), high_net_output], axis=1)
            info_low = layers.fully_connected(info_concat, 32, activation_fn=None,
                                              scope='info_low')

            full_concat = tf.concat([layers.flatten(mconv2), info_low], axis=1)
            full_feature_low = layers.fully_connected(full_concat, 128, activation_fn=tf.nn.relu,
                                                      scope='full_feature_low')
        with tf.variable_scope('actor_low_atk'):
            actor_hidden_low = layers.fully_connected(full_feature_low, 256, activation_fn=tf.nn.relu,
                                                      scope='actor_hidden_low_1')
            actor_hidden_low_2 = layers.fully_connected(actor_hidden_low, 1024, activation_fn=tf.nn.relu,
                                                        scope='actor_hidden_low_2')
            action_low_prob_atk = tf.layers.dense(actor_hidden_low_2, 4096, activation=tf.nn.softmax, kernel_initializer=w_init,
                                         name='action_low_prob')
        with tf.variable_scope('critic_low_atk'):
            critic_hidden_low = layers.fully_connected(full_feature_low, 32, activation_fn=tf.nn.relu,
                                                       scope='critic_hidden_low')
            value_low_atk = layers.fully_connected(critic_hidden_low, 1, activation_fn=None, scope='value_low')
    actor_params_low = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='network_low_atk/actor_low_atk')
    critic_params_low = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='network_low_atk/critic_low_atk')
    feature_params_low = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='network_low_atk/feature_low_atk')
    # 可以在此处重新定义更新参数的方法（考虑为feature_high单独设计梯度）
    a_params_low = actor_params_low + feature_params_low
    c_params_low = critic_params_low + feature_params_low

    return action_low_prob_atk, value_low_atk, a_params_low, c_params_low