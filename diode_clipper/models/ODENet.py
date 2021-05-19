from torch import nn


class ODENetDerivative(nn.Module):
    def __init__(self):
        super().__init__()
        self.densely_connected_layers = nn.Sequential(nn.Linear(2, 4, bias=False), nn.Tanh(), nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 4, bias=False), nn.Tanh(), nn.Linear(4, 1, bias=False))

    def forward(self, t, x):
        return self.densely_connected_layers(x)
