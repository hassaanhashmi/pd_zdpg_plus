import torch.nn as nn


class Actor(nn.Module):
    r"""
    Parameterized Policy φ with scaled NN sizes
    """

    def __init__(self, num_inputs, nn_scale):
        if nn_scale <=0: nn_scale=1
        super(Actor, self).__init__()
        self.num_inputs = num_inputs
        self.fc1 = nn.Linear(self.num_inputs, 32*nn_scale).double()
        self.fc2 = nn.Linear(32*nn_scale, 16*nn_scale).double()
        self.fc3 = nn.Linear(16*nn_scale, self.num_inputs).double()
        self.relu = nn.ReLU().double()
        self.sigmoid = nn.Sigmoid().double()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x