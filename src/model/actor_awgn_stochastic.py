import torch.nn as nn


class NET(nn.Module):
    r"""
    Parameterized stochastic local policy φ_μ^i
    """

    def __init__(self, num_inputs):
        super(NET, self).__init__()
        self.fc1 = nn.Linear(1, 8).double()
        self.fc2 = nn.Linear(8, 4).double()
        self.fc3 = nn.Linear(4, 2).double() #mean and std outputs
        self.relu = nn.ReLU().double()
        self.sigmoid = nn.Sigmoid().double()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x