import warnings
import torch
from torch import nn


class DerivativeMLP(nn.Module):
    def __init__(self, excitation, activation, input_size=2, output_size=1, hidden_size=100):
        super().__init__()
        self.excitation = excitation
        self.densely_connected_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size), activation,
            nn.Linear(hidden_size, hidden_size), activation,
            nn.Linear(hidden_size, output_size))

    def forward(self, t, y):
        """Return the right-hand side of the ODE

        Parameters
        ----------
        t : scalar
            current time point
        y : torch.Tensor of the same shape as the y0 supplied to odeint;
            value of the unknown function at time t

        Returns
        -------
        torch.Tensor of shape the same as y
            derivative of y over time at time t
        """
        excitation = self.excitation(t)

        mlp_input = torch.cat((y, excitation), dim=1)
        output = self.densely_connected_layers(mlp_input)

        return output

    def set_excitation_data(self, time, excitation_data):
        self.excitation.set_excitation_data(time, excitation_data)


class DerivativeMLP2(DerivativeMLP):
    def __init__(self, excitation, activation, input_size=2, output_size=1, hidden_size=100):
        super().__init__(excitation, activation, input_size=2, output_size=1, hidden_size=100)
        self.densely_connected_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size), activation,
            nn.Linear(hidden_size, 2*hidden_size), activation,
            nn.Linear(2*hidden_size, 2*hidden_size), activation,
            nn.Linear(2*hidden_size, 2*hidden_size), activation,
            nn.Linear(2*hidden_size, hidden_size), activation,
            nn.Linear(hidden_size, output_size))


class ODENet(nn.Module):
    def __init__(self, derivative_network, odeint, dt):
        super().__init__()
        self.derivative_network = derivative_network
        self.odeint = odeint
        self.__dt = dt
        self.__true_state = None
        self.time = None
        self.state = None

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
        OUTPUT_FEATURES = 1

        # If there is a ground-truth state provided, use it.
        if self.true_state is not None:
            self.state = self.true_state
        # Else, if there is no state stored from the previous segment computations, initialize the state with 0s.
        elif self.state is None:
            self.state = torch.zeros((minibatch_size, OUTPUT_FEATURES), device=self.device)

        self.create_time_vector(sequence_length)

        self.derivative_network.set_excitation_data(self.time, x)

        odeint_output = self.odeint(self.derivative_network, self.state, self.time)
        # returned tensor is of shape (time_point_count, minibatch_size, other y0 dimensions)

        # Store the last output sample as the initial value for the next segment computation
        self.state = odeint_output[-1]

        return odeint_output

    def create_time_vector(self, sequence_length):
        if self.time is None or self.time.shape[0] != sequence_length:
            self.time = self.dt * torch.arange(0, sequence_length, device=self.device, dtype=torch.double)

    def reset_hidden(self):
        self.__true_state = None
        self.time = None
        self.state = None

    def detach_hidden(self):
        self.state = self.state.detach()

    @property
    def true_state(self):
        return self.__true_state

    @true_state.setter
    def true_state(self, true_state):
        self.__true_state = true_state[1] # First true output sample (check NetworkTraining.true_train_state for details)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dt(self):
        return self.__dt

    @dt.setter
    def dt(self, value):
        self.__dt = value
        self.reset_hidden()
