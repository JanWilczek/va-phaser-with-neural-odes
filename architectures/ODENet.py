import torch
from torch import nn
from CoreAudioML.networks import SimpleRNN


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

    def reset_hidden(self):
        pass

    def detach_hidden(self):
        pass


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
    def __init__(self, excitation, activation, excitation_size=1, output_size=1, hidden_size=30, memory_length=10):
        super().__init__()
        self.excitation = excitation
        self.excitation_size = excitation_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.memory_length = memory_length
        self.memory = None
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

        if self.memory is None:
            self.memory = torch.zeros((y.shape[BATCH_DIMENSION], self.input_size), device=y.device)
        else:
            self.memory = torch.roll(self.memory, shifts=y.shape[FEATURE_DIMENSION], dims=FEATURE_DIMENSION)

        excitation = self.excitation(t)

        assert y.shape[FEATURE_DIMENSION] == self.output_size
        assert excitation.shape[FEATURE_DIMENSION] == self.excitation_size

        mlp_input = torch.cat((y, excitation), dim=FEATURE_DIMENSION)
        self.memory[:, :mlp_input.shape[FEATURE_DIMENSION]] = mlp_input
        output = self.densely_connected_layers(self.memory)

        assert output.shape[FEATURE_DIMENSION] == self.output_size

        return output

    def set_excitation_data(self, time, excitation_data):
        self.excitation.set_excitation_data(time, excitation_data)

    @property
    def input_size(self):
        return (self.memory_length + 1) * (self.excitation_size + self.output_size)

    def reset_hidden(self):
        self.memory = None

    def detach_hidden(self):
        self.memory = self.memory.detach()

class DerivativeLSTM(nn.Module):
    def __init__(self, excitation, activation, excitation_size=1, output_size=1, hidden_size=16):
        super().__init__()
        self.excitation = excitation
        self.excitation_size = excitation_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.rnn = SimpleRNN(self.input_size, self.output_size, unit_type='LSTM', hidden_size=self.hidden_size, skip=0)

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

        rnn_input = torch.cat((y, excitation), dim=FEATURE_DIMENSION).unsqueeze(0)
        output = self.rnn(rnn_input).squeeze(0)

        assert output.shape[FEATURE_DIMENSION] == self.output_size

        return output

    def set_excitation_data(self, time, excitation_data):
        self.excitation.set_excitation_data(time, excitation_data)

    @property
    def input_size(self):
        return self.excitation_size + self.output_size

    def reset_hidden(self):
        self.rnn.reset_hidden()

    def detach_hidden(self):
        self.rnn.detach_hidden()


class ODENet(nn.Module):
    def __init__(self, derivative_network, odeint, dt, target_size):
        """
        Parameters
        ----------
        derivative_network : nn.Module
            [description]
        odeint : [type]
            [description]
        dt : [type]
            [description]
        target_size : int
            Size of the target in the dataset.
            If ODENet's output is just audio (first-order diode clipper, phaser), 
            this number should be 1. 
            If the dataset contains more states than just the audio (e.g., 2 for 
            the second-order diode clipper), put this appropriate number. 
            First target_size elements of the state vector will be returned.
        """
        super().__init__()
        self.derivative_network = derivative_network
        self.odeint = odeint
        self.__dt = dt
        self.target_size = target_size
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
            of shape (x.shape[0], x.shape[1], self.target_size)
        """
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

        odeint_output = self.odeint(self.derivative_network, self.state, self.time, dt=self.dt)
        # returned tensor is of shape (time_point_count, minibatch_size, other y0 dimensions)

        # Store the last output sample as the initial value for the next segment computation
        self.state = odeint_output[-1]

        target_estimate = odeint_output[:, :, :self.target_size]

        assert target_estimate.shape == (x.shape[0], x.shape[1], self.target_size)

        return target_estimate

    def create_time_vector(self, sequence_length):
        if self.time is None or self.time.shape[0] != sequence_length:
            self.time = self.dt * torch.arange(0, sequence_length, device=self.device, dtype=torch.double)

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


class ScaledODENet(ODENet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            must be of shape (sequence_length (e.g., 44100), minibatch_size (no. of sequences in the minibatch), feature_count (e.g., 1 if just an input sample is given))

        Returns
        -------
        output : torch.Tensor
            of shape (x.shape[0], x.shape[1], self.target_size)
        """
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

        odeint_output = self.odeint(lambda t, y: self.derivative_network(t, y * self.dt), self.state, self.time)
        # returned tensor is of shape (time_point_count, minibatch_size, other y0 dimensions)

        # Store the last output sample as the initial value for the next segment computation
        self.state = odeint_output[-1]

        target_estimate = odeint_output[:, :, :self.target_size] * self.dt

        assert target_estimate.shape == (x.shape[0], x.shape[1], self.target_size)

        return target_estimate

    def create_time_vector(self, sequence_length):
        if self.time is None or self.time.shape[0] != sequence_length:
            self.time = torch.arange(0, sequence_length, device=self.device, dtype=torch.double)
