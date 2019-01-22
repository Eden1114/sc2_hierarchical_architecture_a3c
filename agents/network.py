from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers

# DHN add:
def build_high_net(minimap, screen, info, num_macro_action):
    # Extract features
    with tf.variable_scope('actor_high'):
        mconv1_a = layers.conv2d(tf.transpose(minimap, [0, 2, 3, 1]),
                                 num_outputs=16,
                                 kernel_size=5,
                                 stride=1,
                                 scope='mconv1_high')
        mconv2_a = layers.conv2d(mconv1_a,
                                 num_outputs=32,
                                 kernel_size=3,
                                 stride=1,
                                 scope='mconv2_high')
        sconv1_a = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]),
                                 num_outputs=16,
                                 kernel_size=5,
                                 stride=1,
                                 scope='sconv1_high')
        sconv2_a = layers.conv2d(sconv1_a,
                                 num_outputs=32,
                                 kernel_size=3,
                                 stride=1,
                                 scope='sconv2_high')
        info_fc_a = layers.fully_connected(layers.flatten(info),
                                           num_outputs=256,
                                           activation_fn=tf.tanh,
                                           scope='info_fc_high')

        feat_fc_a = tf.concat(
            [layers.flatten(mconv2_a), layers.flatten(sconv2_a), info_fc_a], axis=1)
        feat_fc_a = layers.fully_connected(feat_fc_a,
                                           num_outputs=256,
                                           activation_fn=tf.nn.relu,
                                           scope='feat_fc_high')
        dir_high = layers.fully_connected(feat_fc_a,
                                          num_outputs=num_macro_action,
                                          activation_fn=tf.nn.softmax,
                                          scope='dir_high')

    with tf.variable_scope('critic_high'):
        mconv1_c = layers.conv2d(tf.transpose(minimap, [0, 2, 3, 1]),
                                 num_outputs=16,
                                 kernel_size=5,
                                 stride=1,
                                 scope='mconv1_high')
        mconv2_c = layers.conv2d(mconv1_c,
                                 num_outputs=32,
                                 kernel_size=3,
                                 stride=1,
                                 scope='mconv2_high')
        sconv1_c = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]),
                                 num_outputs=16,
                                 kernel_size=5,
                                 stride=1,
                                 scope='sconv1_high')
        sconv2_c = layers.conv2d(sconv1_c,
                                 num_outputs=32,
                                 kernel_size=3,
                                 stride=1,
                                 scope='sconv2_high')
        info_fc_c = layers.fully_connected(layers.flatten(info),
                                           num_outputs=256,
                                           activation_fn=tf.tanh,
                                           scope='info_fc_high')

        feat_fc_c = tf.concat(
            [layers.flatten(mconv2_c), layers.flatten(sconv2_c), info_fc_c], axis=1)
        feat_fc_c = layers.fully_connected(feat_fc_c,
                                           num_outputs=256,
                                           activation_fn=tf.nn.relu,
                                           scope='feat_fc_high')
        value_high = tf.reshape(layers.fully_connected(feat_fc_c,
                                                       num_outputs=1,
                                                       activation_fn=tf.tanh,
                                                       scope='value_high'), [-1])

    a_params_high = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_high')
    c_params_high = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_high')
    print('a_params_high============', a_params_high)
    print('c_params_high============', c_params_high)
    return dir_high, value_high, a_params_high, c_params_high


def build_low_net(minimap, screen, info, dir_high, act_id):
     # Extract features
     with tf.variable_scope('actor_low'):
          mconv1_a = layers.conv2d(tf.transpose(minimap, [0, 2, 3, 1]),
                                   num_outputs=16,
                                   kernel_size=5,
                                   stride=1,
                                   scope='mconv1_low')
          mconv2_a = layers.conv2d(mconv1_a,
                                   num_outputs=32,
                                   kernel_size=3,
                                   stride=1,
                                   scope='mconv2_low')
          sconv1_a = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]),
                                   num_outputs=16,
                                   kernel_size=5,
                                   stride=1,
                                   scope='sconv1_low')
          sconv2_a = layers.conv2d(sconv1_a,
                                   num_outputs=32,
                                   kernel_size=3,
                                   stride=1,
                                   scope='sconv2_low')

          high_net_output = tf.concat([dir_high, act_id], axis=1)

          high_net_output = layers.fully_connected(high_net_output,
                                                   num_outputs=2,
                                                   activation_fn=tf.tanh,
                                                   scope='high_net_output_low')

          info_a = tf.concat([layers.flatten(info), high_net_output], axis=1)

          info_fc_a = layers.fully_connected(info_a,
                                             num_outputs=256,
                                             activation_fn=tf.tanh,
                                             scope='info_fc_low')

          feat_fc_a = tf.concat(
              [layers.flatten(mconv2_a), layers.flatten(sconv2_a), info_fc_a], axis=1)
          feat_fc_a = layers.fully_connected(feat_fc_a,
                                             num_outputs=256,
                                             activation_fn=tf.nn.relu,
                                             scope='feat_fc_low')
          dir_low = layers.fully_connected(feat_fc_a,
                                           num_outputs=4096,
                                           activation_fn=tf.nn.softmax,
                                           scope='dir_low')

     with tf.variable_scope('critic_low'):
          mconv1_c = layers.conv2d(tf.transpose(minimap, [0, 2, 3, 1]),
                                   num_outputs=16,
                                   kernel_size=5,
                                   stride=1,
                                   scope='mconv1_low')
          mconv2_c = layers.conv2d(mconv1_c,
                                   num_outputs=32,
                                   kernel_size=3,
                                   stride=1,
                                   scope='mconv2_low')
          sconv1_c = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]),
                                   num_outputs=16,
                                   kernel_size=5,
                                   stride=1,
                                   scope='sconv1_low')
          sconv2_c = layers.conv2d(sconv1_c,
                                   num_outputs=32,
                                   kernel_size=3,
                                   stride=1,
                                   scope='sconv2_low')

          high_net_output = tf.concat([dir_high, act_id],axis=1)

          high_net_output = layers.fully_connected(high_net_output,
                                                num_outputs=2,
                                                activation_fn=tf.tanh,
                                                scope='high_net_output_low')

          info_c = tf.concat([layers.flatten(info), high_net_output], axis=1)

          info_fc_c = layers.fully_connected(layers.flatten(info),
                                             num_outputs=256,
                                             activation_fn=tf.tanh,
                                             scope='info_fc_low')

          feat_fc_c = tf.concat(
              [layers.flatten(mconv2_c), layers.flatten(sconv2_c), info_fc_c], axis=1)
          feat_fc_c = layers.fully_connected(feat_fc_c,
                                             num_outputs=256,
                                             activation_fn=tf.nn.relu,
                                             scope='feat_fc_low')
          value_low = tf.reshape(layers.fully_connected(feat_fc_c,
                                                        num_outputs=1,
                                                        activation_fn=tf.tanh,
                                                        scope='value_low'), [-1])

     a_params_low = tf.get_collection(
         tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_low')
     c_params_low = tf.get_collection(
         tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_low')
     print('a_params_low============', a_params_low)
     print('c_params_low============', c_params_low)
     return dir_low, value_low, a_params_low, c_params_low
