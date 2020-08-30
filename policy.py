import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, observation_space, action_space, hidden, learning_rate, gamma):
        super(Policy, self).__init__()
        self.data = []
        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden = hidden
        self.fc1 = nn.Linear(self.observation_space, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.action_space)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.gamma = gamma
		
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)

    def get_reward(self, state):
        if state >= 0.5:
            return 10
        elif state > -0.4:
            return (1.0+state)**2
        else:
            return 0

    def train(self, device):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + self.gamma * R
            loss = -torch.log(prob).to(device) * R
            loss.backward()
		
        self.optimizer.step()
        self.data = []
        return loss.item()
