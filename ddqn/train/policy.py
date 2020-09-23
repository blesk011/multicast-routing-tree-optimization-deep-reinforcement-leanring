import os
import glob
import shutil
from collections import deque

import numpy as np
import tensorflow as tf


class DQN:

    def __init__(self, session, network_size, input_size, output_size, topology_name, name="main"):
        self.session = session
        self.network_size = network_size
        self.input_size = input_size
        self.output_size = output_size
        self.topology_name = topology_name
        self.adj_mat = np.loadtxt("./topology/" + self.topology_name, dtype=int, delimiter=",")
        self.net_name = name
        self._build_network()
        self.saver = tf.train.Saver()

    def _build_network(self, h_size=100, l_rate=0.0001):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")

            W1 = tf.get_variable("W1", shape=[self.input_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            layer1 = tf.nn.relu(tf.matmul(self._X, W1))

            W2 = tf.get_variable("W2", shape=[h_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            layer2 = tf.nn.relu(tf.matmul(layer1, W2))

            W3 = tf.get_variable("W3", shape=[h_size, self.output_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            self._Qpred = tf.matmul(layer2, W3)

        self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

        # Loss function
        loss_bias = self.adj_mat
        loss_bias = np.reshape(loss_bias, [1, self.input_size])
        loss_data = self._Y - self._Qpred
        loss_data = loss_data * loss_bias
        self._loss = tf.reduce_mean(tf.square(loss_data))

        # Learning
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        q_pred = self.session.run(self._Qpred, feed_dict={self._X: x})
        return q_pred

    def preprocess_predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        q_pred = self.session.run(self._Qpred, feed_dict={self._X: x})

        possible_link_bias = self.extract_possible_link(state)
        possible_link_bias = np.reshape(possible_link_bias, [1, self.input_size])

        # ReLu
        # possible_link_bias[possible_link_bias == 0] = 0
        # possible_link_bias[possible_link_bias == 1] = 1

        q_pred = q_pred * possible_link_bias

        return q_pred

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})

    def check_save(self, episode):
        checkpoint_path = self.saver.save(self.session, './model' + '/test', episode)
        print("ckpt files : ", checkpoint_path)

    def load_save(self):
        checkpoint_path = self.saver.restore(self.session, './check/test-199000')
        print("checkpoint files : ", checkpoint_path)

    def extract_possible_link(self, state):
        current_state = state
        selected_node = set(np.where(current_state == 1)[0])
        selected_node = list(selected_node)

        possible_link = np.zeros([self.network_size, self.network_size], dtype=int)
        adjacency_mat = self.adj_mat

        for i in selected_node:
            possible_dst = np.where(adjacency_mat[i] == 1)
            for j in possible_dst[0]:
                possible_link[i][j] = 1
                possible_link[j][i] = 1

        selected_link = np.where(current_state == 1)
        for i in range(len(selected_link[0])):
            possible_link[selected_link[0][i]][selected_link[1][i]] = 0

        return possible_link
