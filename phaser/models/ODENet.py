import torch
import torch.nn as nn
import CoreAudioML.networks as networks


class ODENetDerivative(nn.Module):
    def __init__(self):
        super().__init__()

        self.densely_connected_layers = nn.Sequential(
            nn.Linear(3, 8), nn.Tanh(),
            nn.Linear(8, 8), nn.Tanh(),
            nn.Linear(8, 3), nn.Tanh()
        )
        # self.network = networks.SimpleRNN(unit_type="LSTM", hidden_size=4, skip=0, input_size=3, output_size=3)

    def forward(self, t, x):
        out = self.densely_connected_layers(x)
        # out = self.network(x.unsqueeze(0)).squeeze(0)
        return out

class ODENet(nn.Module):
    def __init__(self, derivative_network, odeint):
        super().__init__()
        self.derivative_network = derivative_network
        self.odeint = odeint
        self.register_buffer('integration_time', torch.Tensor([0, 1]))
        self.state = None

    def forward(self, x):
        sequence_length, minibatch_size, feature_count = x.shape

        output = torch.zeros((sequence_length, minibatch_size, 1), device=self.device)

        if self.state is None:
            self.state = torch.zeros((minibatch_size, 1), device=self.device)

        for n in range(sequence_length):
            input = torch.cat((x[n, :, :], self.state), dim=1)
            out = self.odeint(self.derivative_network, input, self.integration_time)
            # returned tensor is of shape (time_point_count, minibatch_size, other y0 dimensions)

            self.state[:, 0] = out[1, :, 2]
            output[n, :, 0] = out[1, :, 2]

        # output = self.odeint(self.derivative_network, x.permute(1, 0, 2), self.integration_time)
        # output = output[1, :, :, 0].unsqueeze(2).permute(1, 0, 2)

        return output

    def reset_hidden(self):
        self.state = None
        # self.derivative_network.network.reset_hidden()

    def detach_hidden(self):
        self.state = self.state.detach()
        # self.derivative_network.network.detach_hidden()

    @property
    def device(self):
        return next(self.parameters()).device

