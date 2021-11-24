import torch
from torch import nn


class StateTrajectoryNetwork(nn.Module):
    def __init__(self, training_time_step=1.0):
        super().__init__()
        self.densely_connected_layers = nn.Sequential(
            nn.Linear(2, 4, bias=False), nn.Tanh(), 
            nn.Linear(4, 4), nn.Tanh(), 
            nn.Linear(4, 4, bias=False), nn.Tanh(), 
            nn.Linear(4, 1, bias=False))
        self.state = None
        self.device = 'cpu'
        self.__true_state = None
        self.training_time_step = training_time_step
        self.test_time_step = self.training_time_step

    def forward(self, x):
        sequence_length, minibatch_size, feature_count = x.shape
        x = x.permute(1, 0, 2)

        output = torch.zeros((minibatch_size, sequence_length, self.state_size), device=x.device)

        if self.state is None:
            self.state = torch.zeros((minibatch_size, 1, self.state_size), device=self.device)
            
        if self.true_state is not None:
            self.state[:, 0, :] = self.true_state[:, 0, :]

        for n in range(sequence_length):
            # if self.true_state is not None:
                # self.state[:, 0, :] = self.true_state[:, n, :]

            mlp_input = torch.cat((x[:, n:n+1, :], self.state), dim=2)

            # MLPs
            dense_output = self.densely_connected_layers(mlp_input)

            # Residual connection
            output[:, n, :] = self.residual_scaling * dense_output[:, 0, :] + self.state[:, 0, :]

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

    @property
    def dt(self):
        return self.test_time_step

    @dt.setter
    def dt(self, value):
        self.test_time_step = value

    @property
    def residual_scaling(self):
        AMPLITUDE_RANGE = 2.0
        return AMPLITUDE_RANGE * self.test_time_step / self.training_time_step

    @property
    def state_size(self):
        return self.densely_connected_layers[-1].out_features


class FlexibleStateTrajectoryNetwork(StateTrajectoryNetwork):
    def __init__(self, layer_sizes, activation=nn.Tanh(), training_time_step=1.0):
        super().__init__(training_time_step)
        # gain = torch.nn.init.calculate_gain(type(activation).__name__.lower()) # PyTorch's recommended gain calculation
        gain = 0.00001 # This is an arbitrary gain chosen empirically
        layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            linear_layer = nn.Linear(in_size, out_size)
            torch.nn.init.xavier_uniform_(linear_layer.weight, gain)
            with torch.no_grad():
                linear_layer.bias.fill_(0.)
            layers.append(linear_layer)
            layers.append(activation)
        # layers.pop(-1) # Remove the last nonlinearity to have a weighting layer at the output
        self.densely_connected_layers = nn.Sequential(*layers)
        
    @property
    def state_size(self):
        return self.densely_connected_layers[-2].out_features
