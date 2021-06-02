import torch
from torch import nn
from torchinterp1d import Interp1d
from .solvers import forward_euler


class DiodeEquationParameters:
    def __init__(self):
        # From "Numerical Methods for Simulation of Guitar Distortion Circuits" by Yeh et al.
        self.R = 2.2e3
        self.C = 10e-9
        self.i_s = 2.52e-9
        self.v_t = 45.3e-3

p = DiodeEquationParameters()

def diode_equation_rhs(t, v_out, v_in):
    return (v_in - v_out) / (p.R * p.C) - 2 * p.i_s / p.C * torch.sinh(v_out / p.v_t)

class ODENetDerivative(nn.Module):
    def __init__(self):
        super().__init__()
        output_scaling = nn.Linear(1, 1, bias=False)
        output_scaling.weight.data.fill_(44100)
        self.densely_connected_layers = nn.Sequential(
            nn.Linear(2, 4, bias=False), nn.Tanh(), 
            nn.Linear(4, 4, bias=False), nn.Tanh(), 
            nn.Linear(4, 4, bias=False), nn.Tanh(), 
            nn.Linear(4, 1, bias=False),
            output_scaling)
        self.t = None
        self.input = None   # Tensor of shape time_frames x batch_size

    def forward(self, t, y):
        """Return the right-hand side of the ODE

        Parameters
        ----------
        t : scalar
            current time point
        y : torch.Tensor of the same shape as the y0 supplied to odeint
            value of the unknown function at time t

        Returns
        -------
        torch.Tensor of shape the same as y
            derivative of y over time at time t
        """
        # 1st-order interpolation
        # t_new = torch.tile(t.unsqueeze(0), (y.shape[0], 1))
        # input_at_t = Interp1d()(self.t, self.input, t_new)

        # 0th-order interpolation
        frames_count = self.input.shape[0]
        input_at_t = self.input[int(t / self.dt) % frames_count, :]

        # mlp_input = torch.stack((y.clone(), input_at_t, t * torch.ones_like(y)), dim=1)
        mlp_input = torch.stack((y.clone(), input_at_t), dim=1)
        output = self.densely_connected_layers(mlp_input)
        
        # Analytical RHS (not neural) for debugging
        # ode_eq_output = diode_equation_rhs(t, y, input_at_t)

        return output.squeeze_(-1)


class ODENet(nn.Module):
    def __init__(self, derivative_network, odeint=forward_euler, dt=1.0):
        super().__init__()
        self.derivative_network = derivative_network
        self.odeint = odeint
        self.dt = dt
        self.derivative_network.dt = dt
        self.__true_state = None
        self.time = None

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            must be of shape (sequence_length (e.g., 44100), minibatch_size (no. of sequences in the minibatch), feature_count (e.g., 1 if just an input sample is given))

        Returns
        -------
        output : torch.Tensor
            exactly the same shape as x
        """
        sequence_length, minibatch_size, feature_count = x.shape

        self.update_time(sequence_length, minibatch_size)

        self.derivative_network.input = x.squeeze(2)

        initial_value = torch.zeros((minibatch_size,), device=self.device)

        odeint_output = self.odeint(self.derivative_network, initial_value, self.time)
        # returned tensor is of shape (time_point_count, minibatch_size, other y0 dimensions)

        return odeint_output.unsqueeze_(2)

    def update_time(self, sequence_length, minibatch_size):
        if self.time is None:
            start_time = 0.0
        else:
            start_time = self.time[-1] + self.dt
        self.time = torch.linspace(start_time, start_time + sequence_length * self.dt, sequence_length, device=self.device)
        self.derivative_network.t = torch.tile(self.time.unsqueeze(0), (minibatch_size, 1))

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

    @property
    def device(self):
        return next(self.parameters()).device
