import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


class create_CNN(nn.Module):
    def __init__(self, num_action):
        super(create_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_action)

    def forward(self, x, drop):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = F.dropout(x, p=drop)
        x = self.fc2(x)
        return x

class create_LSTM(nn.Module):
    def __init__(self, num_action, train_length, batch_size, cell_size):
        super(create_LSTM, self).__init__()
        self.train_length = train_length
        self.batch_size = batch_size
        self.cell_size = cell_size

        self.conv1 = nn.Conv2d(1, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.conv4 = nn.Conv2d(64, 512, 7, stride=1)
        self.lstm = nn.LSTM(512, 512)
        self.fc = nn.Linear(512, num_action)

    def forward(self, x, drop, state_in, state_out):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x, state_out = self.lstm(x, state_in)
        x = F.dropout(x, p=drop)
        x = self.fc1(x)
        return x, state_out

# def create_LSTM(num_action, num_cell):
#     # network weights
#     W_conv1 = weight_variable([8, 8, 1, 32])
#     b_conv1 = bias_variable([32])
#     W_conv2 = weight_variable([4, 4, 32, 64])
#     b_conv2 = bias_variable([64])
#     W_conv3 = weight_variable([3, 3, 64, 64])
#     b_conv3 = bias_variable([64])
#     W_fc1 = weight_variable([7, 7, 64, 512])
#     b_fc1 = bias_variable([512])
#     W_fc2 = weight_variable([512, num_action])
#     b_fc2 = bias_variable([num_action])
#
#     # training setup
#     trainLength = tf.compat.v1.placeholder(shape=None, dtype=tf.int32)
#
#     # input layer
#     s = tf.compat.v1.placeholder("float", [None, 84, 84, 1])
#     batch_size = tf.compat.v1.placeholder(dtype=tf.int32, shape=[])
#
#     # hidden layers
#     h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
#     h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)
#     h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
#     h_conv4 = tf.nn.relu(conv2d(h_conv3, W_fc1, 1) + b_fc1)
#
#     # define rnn layer
#     rnn_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=num_cell, state_is_tuple=True)
#     convFlat = tf.reshape(tf.compat.v1.layers.flatten(h_conv4), [batch_size, trainLength, num_cell])
#     state_in = rnn_cell.zero_state(batch_size, tf.float32)
#     rnn, rnn_state = tf.nn.dynamic_rnn(
#         inputs=convFlat, cell=rnn_cell, initial_state=state_in)
#     rnn = tf.reshape(rnn, shape=[-1, num_cell])
#
#     drop = tf.compat.v1.placeholder(shape=None, dtype=tf.float32)
#     hidden = tf.nn.dropout(rnn, rate=drop)
#
#     # readout layer
#     readout = tf.matmul(hidden, W_fc2) + b_fc2
#
#     return s, readout, drop, trainLength, batch_size, state_in, rnn_state


class experience_buffer():
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, batch_size, trace_length):
        sampled_episodes = random.sample(self.buffer, batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampledTraces.append(episode[point:point + trace_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces, [batch_size * trace_length, 5])