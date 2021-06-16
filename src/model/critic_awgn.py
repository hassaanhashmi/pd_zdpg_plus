import torch
import torch.nn as nn


class Critic(nn.Module):
    r"""
    Parameterized global critic Q
    """

    def __init__(self, num_states, num_actions, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_states, 4*num_states).double()
        self.fc2 = nn.Linear(4*num_states+num_actions, 4*(num_states+num_actions)).double()
        self.fc3 = nn.Linear(4*(num_states+num_actions), num_states +1).double()
        self.relu = nn.ReLU().double()
    
    def forward(self, xa):
        x, a = xa
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(torch.cat([x,a])))
        x = self.fc3(x)
        return x