import torch
from torch import nn


class StateTrajectoryNetworkFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.densely_connected_layers = nn.Sequential(nn.Linear(2, 4, bias=False), nn.Tanh(), nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 4, bias=False), nn.Tanh(), nn.Linear(4, 1, bias=False))
        self.state = None
        self.device = 'cpu'

    def forward(self, x, true_state=None):
        sequence_length, minibatch_size, feature_count = x.shape
        x = x.permute(1, 0, 2)

        output = torch.zeros_like(x)

        if true_state is not None:
            true_state = true_state.permute(1, 0, 2)
        
        if self.state is None:
            self.state = torch.zeros((minibatch_size, 1, 1), device=self.device)

        for n in range(sequence_length):
            if true_state is not None:
                self.state = true_state[:, n, :].unsqueeze(1)
            
            mlp_input = torch.cat((x[:, n, :].unsqueeze(1), self.state), dim=2)
            
            # MLPs
            dense_output = self.densely_connected_layers(mlp_input)

            # Residual connection
            output[:, n, :] = dense_output[:, 0, :] + self.state[:, 0, :]

            # State update
            if true_state is None:
                self.state[:, 0, :] = output[:, n, :]

        return output.permute(1, 0, 2)

    def reset_hidden(self):
        self.state = None

    def detach_hidden(self):
        self.state = self.state.detach()

    def to(self, device):
        super().to(device)
        self.device = device
