#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/5 9:55
# @Author  : Wang Shuailong

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np


class ResNet(object):
    def __init__(self, board_size, channels, n_outputs, is_training=True, learning_rate=0.001,
                 momentum=0.997, epsilon=1e-5):
        self.board_size = board_size
        self.channels = channels
        self.n_outputs = n_outputs
        self.lr = learning_rate
        self.momentum = momentum
        self.epsilon = epsilon

        self.inputs = tf.placeholder(tf.float32, shape=[None, board_size,
                                                        board_size, channels], name='inputs')
        self.search_probs = tf.placeholder(tf.float32, [None, self.n_outputs], name='search_probs')
        self.search_value = tf.placeholder(tf.float32, [None, 1], name='search_value')

        self.sess = tf.Session()
        self._build_graph(is_training=is_training)
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self, is_training):
        with tf.name_scope('network'):
            self.policy_net, self.value_net = self.build_network(is_training=is_training)

        with tf.name_scope('loss'):
            xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.search_probs,
                                                               logits=self.policy_net)
            value_loss = tf.squared_difference(self.value_net, self.search_value)
            loss_op = tf.reduce_mean(xentropy + value_loss, axis=1)

        with tf.name_scope('train'):
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
            self.train_op = optimizer.minimize(loss_op)

    def build_network(self, is_training):
        resnet_blocks = self.build_resnet(inputs=self.inputs, is_training=is_training)
        policy_net = self.build_policy_net(inputs=resnet_blocks)
        value_net = self.build_value_net(inputs=resnet_blocks)
        return policy_net, value_net

    def build_resnet(self, inputs, is_training):
        with tf.variable_scope('init_conv'):
            net = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=7, activation=None)
            net = tf.layers.batch_normalization(net)
            net = tf.nn.relu(net)
        with tf.variable_scope('max_pooling'):
            net = tf.layers.max_pooling2d(net, pool_size=3, strides=2, padding='SAME')

        with tf.variable_scope('block_1'):
            net = self.residual_block(inputs=net, output_channels=64, is_training=is_training)
            net = self.residual_block(inputs=net, output_channels=64, is_training=is_training)

        with tf.variable_scope('block_2'):
            net = self.residual_block(inputs=net, output_channels=128, is_training=is_training,
                                      same_shape=False)
            net = self.residual_block(inputs=net, output_channels=128, is_training=is_training)

        with tf.variable_scope('output'):
            net = self.batch_normal_relu(net, is_training=is_training)
            net = tf.layers.average_pooling2d(inputs=net, pool_size=2, strides=1)
        return net

    def build_policy_net(self, inputs):
        net_shape = inputs.get_shape().as_list()
        net = tf.reshape(inputs, shape=[-1, np.prod(net_shape[1:])])
        net = tf.layers.dense(net, self.n_outputs, activation=tf.nn.softmax)
        return net

    def build_value_net(self, inputs):
        net_shape = inputs.get_shape().as_list()
        net = tf.reshape(inputs, shape=[-1, np.prod(net_shape[1:])])
        net = tf.layers.dense(net, 1, activation=tf.nn.tanh)
        return net

    def batch_normal_relu(self, inputs, is_training):
        inputs = tf.layers.batch_normalization(inputs, momentum=self.momentum, epsilon=self.epsilon,
                                               training=is_training)
        inputs = tf.nn.relu(inputs)
        return inputs

    def residual_block(self, inputs, output_channels, is_training, same_shape=True):
        strides = 1
        shortcut = inputs
        inputs = self.batch_normal_relu(inputs, is_training)
        if not same_shape:
            strides = 2
            shortcut = tf.layers.conv2d(inputs=inputs, filters=output_channels, kernel_size=1, strides=strides)
        inputs = tf.layers.conv2d(inputs=inputs, filters=output_channels, kernel_size=3,
                                  strides=strides, padding='SAME')
        inputs = self.batch_normal_relu(inputs, is_training)
        inputs = tf.layers.conv2d(inputs=inputs, filters=output_channels, kernel_size=3, strides=1, padding='SAME')

        outputs = shortcut + inputs
        return outputs

    def predict_actions(self, s):
        actions_prob = self.sess.run(self.policy_net, feed_dict={self.inputs: s})[0]
        return actions_prob

    def predict_value(self, s):
        state_value = self.sess.run(self.value_net, feed_dict={self.inputs: s})[0, 0]
        return state_value



x_data = np.random.rand(1, 28, 28, 1).astype('float32')

resnet = ResNet(board_size=28, channels=1, n_outputs=10)

actions_prob = resnet.predict_actions(x_data)
value = resnet.predict_value(x_data)
print('actions prob: ', actions_prob)
print('state value: ', value)


