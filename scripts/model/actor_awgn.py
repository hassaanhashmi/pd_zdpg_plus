import torch.nn as nn


class Actor(nn.Module):
    r"""
    Parameterized deterministic olicy φ^i
    """

    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(1, 8).double()
        self.fc2 = nn.Linear(8, 4).double()
        self.fc3 = nn.Linear(4, 1).double()
        self.relu = nn.ReLU().double()
        self.sigmoid = nn.Sigmoid().double()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x