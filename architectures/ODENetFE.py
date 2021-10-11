import torch
from torch import nn


class DerivativeMLPFE(nn.Module):
    def __init__(self, activation, excitation_size=1, output_size=1, hidden_size=100):
        super().__init__()
        self.excitation_size = excitation_size
        self.output_size = output_size
        self.densely_connected_layers = nn.Sequential(
            nn.Linear(self.input_size, hidden_size), activation,
            nn.Linear(hidden_size, hidden_size), activation,
            nn.Linear(hidden_size, self.output_size))

    def forward(self, t, y):
        BATCH_DIMENSION = 0
        FEATURE_DIMENSION = 1

        excitation = self.excitation[t]

        assert y.shape[FEATURE_DIMENSION] == self.output_size
        assert excitation.shape[FEATURE_DIMENSION] == self.excitation_size

        mlp_input = torch.cat((y, excitation), dim=FEATURE_DIMENSION)
        output = self.densely_connected_layers(mlp_input)

        assert mlp_input.shape[FEATURE_DIMENSION] == self.input_size
        assert output.shape[FEATURE_DIMENSION] == self.output_size

        return output

    @property
    def input_size(self):
        return self.excitation_size + self.output_size

    def reset_hidden(self):
        pass

    def detach_hidden(self):
        pass


class DerivativeMLP2FE(DerivativeMLPFE):
    def __init__(self, activation, excitation_size=1, output_size=1, hidden_size=100):
        super().__init__(activation, excitation_size=excitation_size, output_size=output_size, hidden_size=hidden_size)
        self.densely_connected_layers = nn.Sequential(
            nn.Linear(self.input_size, hidden_size), activation,
            nn.Linear(hidden_size, 2*hidden_size), activation,
            nn.Linear(2*hidden_size, 2*hidden_size), activation,
            nn.Linear(2*hidden_size, 2*hidden_size), activation,
            nn.Linear(2*hidden_size, hidden_size), activation,
            nn.Linear(hidden_size, self.output_size))


class DerivativeFEWithMemory(nn.Module):
    def __init__(self, activation, excitation_size=1, output_size=1, hidden_size=30, memory_length=10):
        super().__init__()
        self.excitation_size = excitation_size
        self.output_size = output_size
        self.memory_length = memory_length
        self.memory = None
        self.densely_connected_layers = nn.Sequential(
            nn.Linear(self.input_size, hidden_size), activation,
            nn.Linear(hidden_size, hidden_size), activation,
            nn.Linear(hidden_size, self.output_size))

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

        if self.memory is None:
            self.memory = torch.zeros((y.shape[BATCH_DIMENSION], self.input_size), device=y.device)
        else:
            self.memory = torch.roll(self.memory, shifts=y.shape[FEATURE_DIMENSION], dims=FEATURE_DIMENSION)

        excitation = self.excitation[t]

        assert y.shape[FEATURE_DIMENSION] == self.output_size
        assert excitation.shape[FEATURE_DIMENSION] == self.excitation_size

        mlp_input = torch.cat((y, excitation), dim=FEATURE_DIMENSION)
        self.memory[:, :mlp_input.shape[FEATURE_DIMENSION]] = mlp_input
        output = self.densely_connected_layers(self.memory)

        assert output.shape[FEATURE_DIMENSION] == self.output_size

        return output

    @property
    def input_size(self):
        return (self.memory_length + 1) * (self.excitation_size + self.output_size)

    def reset_hidden(self):
        self.memory = None

    def detach_hidden(self):
        self.memory = self.memory.detach()


class ScaledODENetFE(nn.Module):
    def __init__(self, derivative_network, sampling_rate):
        super().__init__()
        self.derivative_network = derivative_network
        self.sampling_rate = sampling_rate
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
        self.derivative_network.excitation = x

        y = torch.empty((self.time.shape[0], *self.state.shape), dtype=self.state.dtype, device=self.device)
        y[0] = self.state
        y0 = self.sampling_rate * self.state

        n = 1
        for t0, t1 in zip(self.time[:-1], self.time[1:]):
            y1 = y0 + self.derivative_network(t0, y0 / self.sampling_rate)
            y[n] = y1 / self.sampling_rate
            n += 1
            y0 = y1

        # Store the last output sample as the initial value for the next segment computation
        self.state = y[-1]

        return y[:, :, OUTPUT_FEATURE_ID].unsqueeze(2)

    def create_time_vector(self, sequence_length):
        if self.time is None or self.time.shape[0] != sequence_length:
            self.time = torch.arange(0, sequence_length, device=self.device)

    def reset_hidden(self):
        self.__true_state = None
        self.time = None
        self.state = None
        self.derivative_network.reset_hidden()

    def detach_hidden(self):
        self.state = self.state.detach()
        self.derivative_network.detach_hidden()

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
        return 1 / self.sampling_rate

    @dt.setter
    def dt(self, value):
        self.sampling_rate = int(1 / value)
        self.reset_hidden()

    @property
    def state_size(self):
        return self.derivative_network.output_size

    @property
    def excitation_size(self):
        return self.derivative_network.excitation_size
