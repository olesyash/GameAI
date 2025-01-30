import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


class GameNetwork(nn.Module):
    # def __init__(self,board_size):
    #     super(GameNetwork, self).__init__()
    #     self.board_size = board_size
    #     pass
    #
    #     self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
    #     self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    #     self.conv3 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
    #
    #     #policy head
    #     self.policy_conv = nn.Conv2d(256, 32, kernel_size=1)
    #     self.policy_fc = nn.Linear(32*board_size*board_size, board_size*board_size)
    #
    #     #value head
    #     self.value_conv = nn.Conv2d(256, 32, kernel_size=1)
    #     self.value_fc1 = nn.Linear(32*board_size*board_size, 256)
    #     self.value_fc2 = nn.Linear(256, 1)
    #
    # def forward(self, x):
    #     x = F.relu(self.conv1(x))
    #     x = F.relu(self.conv2(x))
    #     x = F.relu(self.conv3(x))
    #
    #     #policy head
    #     policy = self.policy_conv(x)
    #     policy = policy.view(-1, 32*self.board_size*self.board_size)
    #     policy = self.policy_fc(policy)
    #     policy = F.log_softmax(policy, dim=1)
    #
    #     #value head
    #     value = self.value_conv(x)
    #     value = value.view(-1, 32*self.board_size*self.board_size)
    #     value = F.relu(self.value_fc1(value))
    #     value = torch.tanh(self.value_fc2(value))
    #
    #     return value, policy

    def __init__(self, board_size):
        super(GameNetwork, self).__init__()
        self.board_size = board_size
        input_size = board_size * board_size  # Flattened board

        # Shared layers
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)

        # Policy head
        self.policy_fc1 = nn.Linear(256, 256)
        self.policy_out = nn.Linear(256, board_size * board_size)

        # Value head
        self.value_fc1 = nn.Linear(256, 128)
        self.value_out = nn.Linear(128, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Flatten the input
        x = x.view(-1, self.board_size * self.board_size)

        # Shared layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        # Policy head
        policy = F.relu(self.policy_fc1(x))
        policy = self.dropout(policy)
        policy = self.policy_out(policy)
        policy = F.softmax(policy, dim=1)

        # Value head
        value = F.relu(self.value_fc1(x))
        value = self.dropout(value)
        value = torch.tanh(self.value_out(value))

        return policy, value

    def train(self,board_size, states, targets):
        optimizer = torch.optim.Adam(self.parameters())
        criterion_value = nn.MSELoss()
        criterion_policy = nn.NLLLoss()
        for epoch in range(100):
            for state, target in zip(states, targets):
                state = torch.tensor(state).float().unsqueeze(0)
                target = torch.tensor(target).float().unsqueeze(0)
                value, policy = self(state)
                loss_value = criterion_value(value, target[0])
                loss_policy = criterion_policy(policy, target[1])
                loss = loss_value + loss_policy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    def predict(self, state):

        self
        return V, P
        

    def save_model(self):
        pass

    def load_model(self):
        pass