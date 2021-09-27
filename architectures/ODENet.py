import warnings
import torch
from torch import nn


class DerivativeMLP(nn.Module):
    def __init__(self, excitation, activation, excitation_size=1, output_size=1, hidden_size=100):
        super().__init__()
        self.excitation = excitation
        self.excitation_size = excitation_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.densely_connected_layers = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size), activation,
            nn.Linear(self.hidden_size, self.hidden_size), activation,
            nn.Linear(self.hidden_size, self.output_size))

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
        BATCH_DIMENSION = 0
        FEATURE_DIMENSION = 1

        excitation = self.excitation(t)

        assert y.shape[FEATURE_DIMENSION] == self.output_size
        assert excitation.shape[FEATURE_DIMENSION] == self.excitation_size

        mlp_input = torch.cat((y, excitation), dim=FEATURE_DIMENSION)
        output = self.densely_connected_layers(mlp_input)

        assert mlp_input.shape[FEATURE_DIMENSION] == self.input_size
        assert output.shape[FEATURE_DIMENSION] == self.output_size

        return output

    def set_excitation_data(self, time, excitation_data):
        self.excitation.set_excitation_data(time, excitation_data)

    @property
    def input_size(self):
        return self.excitation_size + self.output_size


class DerivativeMLP2(DerivativeMLP):
    def __init__(self, excitation, activation, excitation_size=1, output_size=1, hidden_size=100):
        super().__init__(excitation, activation, excitation_size=excitation_size, output_size=output_size, hidden_size=hidden_size)
        self.densely_connected_layers = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size), activation,
            nn.Linear(self.hidden_size, 2*self.hidden_size), activation,
            nn.Linear(2*self.hidden_size, 2*self.hidden_size), activation,
            nn.Linear(2*self.hidden_size, 2*self.hidden_size), activation,
            nn.Linear(2*self.hidden_size, self.hidden_size), activation,
            nn.Linear(self.hidden_size, self.output_size))

class SingleLinearLayer(DerivativeMLP):
    def __init__(self, excitation, activation, excitation_size, output_size, hidden_size):
        super().__init__(excitation, activation, excitation_size, output_size, hidden_size)
        self.densely_connected_layers = nn.Sequential(nn.Linear(self.input_size, self.output_size, bias=False), activation)

class ScaledSingleLinearLayer(SingleLinearLayer):
    def __init__(self, excitation, activation, excitation_size, output_size, hidden_size):
        super().__init__(excitation, activation, excitation_size, output_size, hidden_size)
        scaling = nn.Linear(self.output_size, self.output_size, bias=False)
        scaling.weight.data.fill_(441.0)
        # scaling.weight.data.requires_grad_(False)
        self.densely_connected_layers = nn.Sequential(nn.Linear(self.input_size, self.output_size, bias=False), activation, scaling)

class DerivativeWithMemory(nn.Module):
    def __init__(self, excitation, activation, excitation_size=1, output_size=1, hidden_size=10, memory_length=10):
        super().__init__()
        self.excitation = excitation
        self.excitation_size = excitation_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.densely_connected_layers = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size), activation,
            nn.Linear(self.hidden_size, self.hidden_size), activation,
            nn.Linear(self.hidden_size, self.output_size))

    def forward(t, y):
        raise NotImplementedError


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
        # The first element of the state is the audio sample output
        OUTPUT_FEATURE_ID = 0

        sequence_length, minibatch_size, feature_count = x.shape

        # If there is no state stored from the previous segment computations, initialize the state with 0s.
        if self.state is None:
            self.state = torch.zeros((minibatch_size, self.state_size), device=self.device)

        # If there is a ground-truth state provided, use it.
        if self.true_state is not None:
            # If the true state is 1-dimensional, it is just the audio output; preserve the rest of the state for this iteration
            if self.true_state.shape[1] == 1:
                self.state[:, OUTPUT_FEATURE_ID] = self.true_state.squeeze()
            else:
                # If the true state is multidimensional, assign it to the first entries of self.state
                self.state[:, :self.true_state.shape[1]] = self.true_state            

        self.create_time_vector(sequence_length)

        self.derivative_network.set_excitation_data(self.time, x)

        odeint_output = self.odeint(self.derivative_network, self.state, self.time)
        # returned tensor is of shape (time_point_count, minibatch_size, other y0 dimensions)

        # Store the last output sample as the initial value for the next segment computation
        self.state = odeint_output[-1]

        return odeint_output[:, :, OUTPUT_FEATURE_ID].unsqueeze(2)

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

    @property
    def state_size(self):
        return self.derivative_network.output_size

    @property
    def excitation_size(self):
        return self.derivative_network.excitation_size
