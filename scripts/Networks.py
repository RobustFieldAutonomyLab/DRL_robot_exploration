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
    def __init__(self, num_action):
        super(create_LSTM, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.conv4 = nn.Conv2d(64, 512, 7, stride=1)
        self.lstm = nn.LSTM(512, 512, batch_first=True)
        self.fc = nn.Linear(512, num_action)

    def forward(self, x, drop, state_in, train_length, batch_size, cell_size):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = torch.reshape(x.flatten(), (batch_size, train_length, cell_size))
        x, state_out = self.lstm(x, state_in)
        x = torch.reshape(x, (-1, cell_size))
        x = F.dropout(x, p=drop)
        x = self.fc(x)
        return x, state_out
