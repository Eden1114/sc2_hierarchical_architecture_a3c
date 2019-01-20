from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers

# 以下函数不再使用：
def build_fcn(minimap, screen, info, msize, ssize, num_action):
  # Extract features
  mconv1 = layers.conv2d(tf.transpose(minimap, [0, 2, 3, 1]),
                         num_outputs=16,
                         kernel_size=5,
                         stride=1,
                         scope='mconv1')
  mconv2 = layers.conv2d(mconv1,
                         num_outputs=32,
                         kernel_size=3,
                         stride=1,
                         scope='mconv2')
  sconv1 = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]),
                         num_outputs=16,
                         kernel_size=5,
                         stride=1,
                         scope='sconv1')
  sconv2 = layers.conv2d(sconv1,
                         num_outputs=32,
                         kernel_size=3,
                         stride=1,
                         scope='sconv2')
  info_fc = layers.fully_connected(layers.flatten(info),
                                   num_outputs=256,
                                   activation_fn=tf.tanh,
                                   scope='info_fc')

  # Compute spatial actions
  feat_conv = tf.concat([mconv2, sconv2], axis=3)
  spatial_action = layers.conv2d(feat_conv,
                                 num_outputs=1, # 输出并不是1？？？ 不知为何
                                 kernel_size=1,
                                 stride=1,
                                 activation_fn=None,
                                 scope='spatial_action')
  spatial_action = tf.nn.softmax(layers.flatten(spatial_action))


  # Compute non spatial actions and value
  feat_fc = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2), info_fc], axis=1)
  feat_fc = layers.fully_connected(feat_fc,
                                   num_outputs=256,
                                   activation_fn=tf.nn.relu,
                                   scope='feat_fc')
  non_spatial_action = layers.fully_connected(feat_fc,
                                              num_outputs=num_action,
                                              activation_fn=tf.nn.softmax,
                                              scope='non_spatial_action')
  value = tf.reshape(layers.fully_connected(feat_fc,
                                            num_outputs=1,
                                            activation_fn=None,
                                            scope='value'), [-1])

  return spatial_action, non_spatial_action, value


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

        feat_fc_a = tf.concat([layers.flatten(mconv2_a), layers.flatten(sconv2_a), info_fc_a], axis=1)
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

        feat_fc_c = tf.concat([layers.flatten(mconv2_c), layers.flatten(sconv2_c), info_fc_c], axis=1)
        feat_fc_c = layers.fully_connected(feat_fc_c,
                                         num_outputs=256,
                                         activation_fn=tf.nn.relu,
                                         scope='feat_fc_high')
        value_high = tf.reshape(layers.fully_connected(feat_fc_c,
                                                  num_outputs=1,
                                                  activation_fn=tf.tanh,
                                                  scope='value_high'), [-1])

    a_params_high = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_high')
    c_params_high = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_high')
    print('a_params_high============', a_params_high)
    print('c_params_high============', c_params_high)
    return dir_high, value_high, a_params_high, c_params_high


def build_low_net(minimap, screen, info):
  # Extract features
  with tf.variable_scope('actor_low'):
      mconv1_a = layers.conv2d(tf.transpose(minimap, [0, 2, 3, 1]),
                             num_outputs=16,
                             kernel_size=5,
                             stride=1,
                             scope='mconv1_low')
      print("m1 ==", mconv1_a.get_shape())
      mconv2_a = layers.conv2d(mconv1_a,
                             num_outputs=32,
                             kernel_size=3,
                             stride=1,
                             scope='mconv2_low')
      print("m2 ==", mconv2_a.get_shape())
      sconv1_a = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]),
                             num_outputs=16,
                             kernel_size=5,
                             stride=1,
                             scope='sconv1_low')
      print("s1 ==", sconv1_a.get_shape())
      sconv2_a = layers.conv2d(sconv1_a,
                             num_outputs=32,
                             kernel_size=3,
                             stride=1,
                             scope='sconv2_low')
      print("s2 ==", sconv2_a.get_shape())
      # Compute spatial actions
      feat_conv_a = tf.concat([mconv2_a, sconv2_a], axis=3)
      print("d ==", feat_conv_a.get_shape())
      spatial_action_low = layers.conv2d(feat_conv_a,
                                     num_outputs=1,
                                     kernel_size=1,
                                     stride=1,
                                     activation_fn=tf.tanh,  # 从None改为tanh，即将值限制在-1到1
                                     scope='spatial_action_low')
      print("dd ==", spatial_action_low.get_shape())
      spatial_action_low = tf.nn.softmax(layers.flatten(spatial_action_low))
      print("dddd ==", spatial_action_low.get_shape())

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
      info_fc_c = layers.fully_connected(layers.flatten(info),
                                       num_outputs=256,
                                       activation_fn=tf.tanh,
                                       scope='info_fc_low')
      # Compute value
      feat_fc_c = tf.concat([layers.flatten(mconv2_c), layers.flatten(sconv2_c), info_fc_c], axis=1)
      feat_fc_c = layers.fully_connected(feat_fc_c,
                                       num_outputs=256,
                                       activation_fn=tf.nn.relu,
                                       scope='feat_fc_low')

      value_low = tf.reshape(layers.fully_connected(feat_fc_c,
                                                    num_outputs=1,
                                                    activation_fn=tf.tanh,  # 从None改为tanh，即将值限制在-1到1
                                                    scope='value_low'), [-1])

  a_params_low = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_low')
  c_params_low = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_low')
  print('a_params_low============', a_params_low)
  print('c_params_low============', c_params_low)
  return spatial_action_low, value_low, a_params_low, c_params_low
