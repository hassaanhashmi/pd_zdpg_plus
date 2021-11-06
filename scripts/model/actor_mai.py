import torch.nn as nn


class Actor(nn.Module):
    r"""
    Parameterized Deterministic global policy φ
    """

    def __init__(self, num_inputs):
        super(Actor, self).__init__()
        self.num_inputs = num_inputs
        self.fc1 = nn.Linear(self.num_inputs, 64).double()
        self.fc2 = nn.Linear(64, 32).double()
        self.fc3 = nn.Linear(32, self.num_inputs).double()
        self.relu = nn.ReLU().double()
        self.sigmoid = nn.Sigmoid().double()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x