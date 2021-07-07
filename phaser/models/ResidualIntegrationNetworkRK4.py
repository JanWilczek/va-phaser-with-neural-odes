import torch
import torch.nn as nn


class BilinearBlock(nn.Module):
    def __init__(self, input_size=1, output_size=1, latent_size=6):
        super().__init__()
        self.fc1 = nn.Linear(input_size, latent_size)
        self.nl1 = nn.Tanh()
        self.fc2 = nn.Linear(input_size, latent_size)
        self.nl2 = nn.Tanh()
        self.fc3 = nn.Linear(input_size, latent_size)
        self.nl3 = nn.Tanh()
        self.fc4 = nn.Linear(2 * latent_size, output_size)
        self.nl4 = nn.Tanh()

    def forward(self, x):
        fc1_out = self.nl1(self.fc1(x))
        fc2_out = self.nl2(self.fc2(x))
        fc3_out = self.nl3(self.fc3(x))
        fc4_input = torch.cat((fc1_out, torch.mul(fc2_out, fc3_out)), dim=1)
        out = self.nl4(self.fc4(fc4_input))
        return out

class ResidualIntegrationNetworkRK4(nn.Module):
    def __init__(self, derivative_network, dt=1.0):
        super().__init__()
        self.derivative_network = derivative_network
        self.dt = dt
        self.__true_state = None

    def forward(self, x):
        sequence_length, minibatch_size, feature_count = x.shape
        OUTPUT_FEATURES = 1

        output = torch.empty((sequence_length, minibatch_size, OUTPUT_FEATURES), device=x.device)

        # Explicit Runge-Kutta scheme of order 4
        y0 = torch.zeros((minibatch_size, OUTPUT_FEATURES), device=x.device)

        for time_frame_id in range(sequence_length):
            # Retrieve the correct last output if available
            if self.true_state is not None:
                y0 = self.true_state[time_frame_id]

            v_in = x[time_frame_id]

            TIME_FRAME_FEATURE_DIMENSION = 1
            k1 = self.derivative_network(torch.cat((v_in, y0), dim=TIME_FRAME_FEATURE_DIMENSION))
            k2 = self.derivative_network(torch.cat((v_in, y0+k1*self.dt/2), dim=TIME_FRAME_FEATURE_DIMENSION))
            k3 = self.derivative_network(torch.cat((v_in, y0+k2*self.dt/2), dim=TIME_FRAME_FEATURE_DIMENSION))
            k4 = self.derivative_network(torch.cat((v_in, y0+k3*self.dt), dim=TIME_FRAME_FEATURE_DIMENSION))

            output[time_frame_id] = y0 + 1/6 * (k1 + 2*k2 + 2*k3 + k4) * self.dt
            y0 = output[time_frame_id]

        return output

    def reset_hidden(self):
        self.__true_state = None

    def detach_hidden(self):
        pass

    @property
    def true_state(self):
        return self.__true_state

    @true_state.setter
    def true_state(self, true_state):
        self.__true_state = true_state
