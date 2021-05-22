import torch
from torch import nn


def forward_euler(f, y0, t):
    y = torch.zeros((t.shape[0], y0.shape[0]), device=t.device)
    y[0, :] = y0
    dt = t[1] - t[0]  # assume equidistant sampling

    for n in range(y.shape[0]):
        y[n + 1, :] = y[n, :] + dt * f(t, y[n, :])

    return y


class ODENetDerivative(nn.Module):
    def __init__(self):
        super().__init__()
        self.densely_connected_layers = nn.Sequential(
            nn.Linear(
                2, 4, bias=False), nn.Tanh(), nn.Linear(
                4, 4), nn.Tanh(), nn.Linear(
                4, 4, bias=False), nn.Tanh(), nn.Linear(
                    4, 1, bias=False))

    def forward(self, t, y):
        return self.densely_connected_layers(y)


class ODENet(nn.Module):
    def __init__(self, derivative_network, odeint=forward_euler):
        super().__init__()
        self.derivative_network = derivative_network
        self.odeint = odeint
        self.state = None
        self.__true_state = None

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            must be of shape (sequence_length (e.g., 44100), minibatch_size (no. of sequences in the minibatch), feature_count (e.g., 1 if just an input samle is given))

        Returns
        -------
        output : torch.Tensor
            exactly the same shape as x
        """
        sequence_length, minibatch_size, feature_count = x.shape
        x = x.permute(1, 0, 2)

        device = next(self.parameters()).device

        output = torch.zeros_like(x)

        if self.state is None:
            self.state = torch.zeros((minibatch_size, 1, 1), device=device)

        for n in range(sequence_length):
            if self.true_state is not None:
                self.derivative_network.state[:, 0, :] = self.true_state[:, n, :]

            initial_value = torch.cat((x[:, n, :].unsqueeze(1), self.state), dim=2)
            # initial_value = self.state

            time = torch.Tensor([n - 1, n]).to(device)

            odeint_output = self.odeint(self.derivative_network, initial_value, time)

            # take the second feature? and the second time step value, i.e., t[1]
            output[:, n, :] = odeint_output[1, :, :, 1]

            # State update
            self.state[:, 0, :] = output[:, n, :]

        return output.permute(1, 0, 2)

    def reset_hidden(self):
        self.state = None
        self.true_state = None

    def detach_hidden(self):
        self.state = self.state.detach()

    @property
    def true_state(self):
        return self.__true_state

    @true_state.setter
    def true_state(self, true_state):
        if true_state is None:
            self.__true_state = true_state
        else:
            self.__true_state = true_state.permute(1, 0, 2)
