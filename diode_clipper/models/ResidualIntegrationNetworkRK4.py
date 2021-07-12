import torch
import torch.nn as nn


class BilinearBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 6)
        self.nl1 = nn.Tanh()
        self.fc2 = nn.Linear(2, 6)
        self.nl2 = nn.Tanh()
        self.fc3 = nn.Linear(2, 6)
        self.nl3 = nn.Tanh()
        self.fc4 = nn.Linear(12, 1)
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
        self.true_state = None

    def forward(self, x):
        sequence_length, minibatch_size, feature_count = x.shape

        output = torch.empty((sequence_length, minibatch_size, 1), device=x.device)

        # Explicit Runge-Kutta scheme of order 4
        y0 = torch.zeros((minibatch_size, 1), device=x.device)

        for n in range(sequence_length):
            # Retrieve the correct last output if available
            if self.true_state is not None:
                y0 = self.true_state[n]

            v_in = x[n]

            k1 = self.derivative_network(torch.cat((v_in, y0), dim=1))
            k2 = self.derivative_network(torch.cat((v_in, y0+k1*self.dt/2), dim=1))
            k3 = self.derivative_network(torch.cat((v_in, y0+k2*self.dt/2), dim=1))
            k4 = self.derivative_network(torch.cat((v_in, y0+k3*self.dt), dim=1))

            output[n] = y0 + 1/6 * (k1 + 2*k2 + 2*k3 + k4) * self.dt
            y0 = output[n]

        return output

    def reset_hidden(self):
        self.true_state = None

    def detach_hidden(self):
        pass

    @property
    def true_state(self):
        return self.__true_state

    @true_state.setter
    def true_state(self, true_state):
        self.__true_state = true_state

    @property
    def dt(self):
        return 1.0  # Enforce time step 1

    @dt.setter
    def dt(self, value):
        pass
