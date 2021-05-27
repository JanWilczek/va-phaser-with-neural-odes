import torch
from torch import nn
from torchinterp1d import Interp1d
from .solvers import forward_euler


class ODENetDerivative(nn.Module):
    def __init__(self):
        super().__init__()
        self.densely_connected_layers = nn.Sequential(
            nn.Linear(
                2, 4, bias=False), nn.Tanh(), nn.Linear(
                4, 4), nn.Tanh(), nn.Linear(
                4, 4, bias=False), nn.Tanh(), nn.Linear(
                    4, 1, bias=False))
        self.t = None
        self.input = None

    def forward(self, t, y):
        """Return the right-hand side of the ODE

        Parameters
        ----------
        t : scalar
            current time point
        y : torch.Tensor of shape (minibatch_size, 1, 1)
            value of the unknown function at time t

        Returns
        -------
        torch.Tensor of shape (minibatch_size, 1, 1)
            derivative of y over time at time t
        """
        t_new = torch.tile(t.unsqueeze(0), (y.shape[0], 1))
        input_at_t = Interp1d()(self.t, self.input, t_new).unsqueeze(-1)
        mlp_input = torch.cat((y.unsqueeze(1).unsqueeze(2), input_at_t), dim=2)
        output = self.densely_connected_layers(mlp_input)
        return output.squeeze()


class ODENet(nn.Module):
    def __init__(self, derivative_network, odeint=forward_euler, dt=1.0, method='euler'):
        super().__init__()
        self.derivative_network = derivative_network
        self.odeint = odeint
        self.dt = dt
        self.method = method
        self.__true_state = None
        self.time = None

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

        if self.time is None:
            start_time = 0.0
        else:
            start_time = self.time[-1] + self.dt
        self.time = torch.linspace(start_time, start_time + sequence_length * self.dt, sequence_length, device=device)
        self.derivative_network.t = torch.tile(self.time.unsqueeze(0), (minibatch_size, 1))

        self.derivative_network.input = x.squeeze()

        initial_value = torch.zeros((minibatch_size,), device=device)

        odeint_output = self.odeint(self.derivative_network, initial_value, self.time, method=self.method)
        # returned tensor is of shape (time_point_count, minibatch_size, 1, features_count (=1 here))

        return odeint_output.unsqueeze(1)

    def reset_hidden(self):
        self.true_state = None
        self.time = None

    def detach_hidden(self):
        pass

    @property
    def true_state(self):
        return self.__true_state

    @true_state.setter
    def true_state(self, true_state):
        if true_state is None:
            self.__true_state = true_state
        else:
            self.__true_state = true_state.permute(1, 0, 2)

    def extra_repr(self):
        return f'method={self.method};'
