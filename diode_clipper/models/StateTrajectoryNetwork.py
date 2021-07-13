import torch
from torch import nn


class StateTrajectoryNetworkFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.densely_connected_layers = nn.Sequential(
            nn.Linear(2, 4, bias=False), nn.Tanh(), 
            nn.Linear(4, 4), nn.Tanh(), 
            nn.Linear(4, 4, bias=False), nn.Tanh(), 
            nn.Linear(4, 1, bias=False))
        self.state = None
        self.device = 'cpu'
        self.__true_state = None

    def forward(self, x):
        sequence_length, minibatch_size, feature_count = x.shape
        x = x.permute(1, 0, 2)

        output = torch.zeros_like(x)

        if self.state is None:
            self.state = torch.zeros((minibatch_size, 1, 1), device=self.device)

        for n in range(sequence_length):
            if self.true_state is not None:
                self.state[:, 0, :] = self.true_state[:, n, :]

            mlp_input = torch.cat((x[:, n, :].unsqueeze(1), self.state), dim=2)

            # MLPs
            dense_output = self.densely_connected_layers(mlp_input)

            # Residual connection
            output[:, n, :] = dense_output[:, 0, :] + self.state[:, 0, :]

            # State update
            self.state[:, 0, :] = output[:, n, :]

        return output.permute(1, 0, 2)

    def reset_hidden(self):
        self.state = None
        self.true_state = None

    def detach_hidden(self):
        self.state = self.state.detach()

    def to(self, device):
        super().to(device)
        self.device = device

    @property
    def true_state(self):
        return self.__true_state

    @true_state.setter
    def true_state(self, true_state):
        if true_state is None:
            self.__true_state = true_state
        else:
            self.__true_state = true_state.permute(1, 0, 2)
