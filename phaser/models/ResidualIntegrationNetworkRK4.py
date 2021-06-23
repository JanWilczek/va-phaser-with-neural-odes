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
        self.fc4 = nn.Linear(2 * latent_size, 1)
        self.nl4 = nn.Tanh()

    def forward(self, x):
        fc1_out = self.nl1(self.fc1(x))
        fc2_out = self.nl2(self.fc2(x))
        fc3_out = self.nl3(self.fc3(x))
        fc4_input = torch.cat((fc1_out, torch.mul(fc2_out, fc3_out)), dim=1)
        out = self.nl4(self.fc4(fc4_input))
        return out

class ResidualIntegrationNetworkRK4(nn.Module):
    def __init__(self, derivative_network):
      super().__init__()
      self.derivative_network = derivative_network

    def forward(self, x):
        sequence_length, batch_size, feature_count = x.shape

        output = torch.zeros((sequence_length, batch_size, 1), device=x.device)

        # Explicit Runge-Kutta scheme of order 4
        # Time step is assumed to be 1, i.e., dt=1
        y0 = torch.zeros((batch_size, 1), device=x.device)

        for n in range(sequence_length):
            v_in = x[n, :, :]

            k1 = self.derivative_network(torch.cat((v_in, y0), dim=1))
            k2 = self.derivative_network(torch.cat((v_in, y0+k1/2), dim=1))
            k3 = self.derivative_network(torch.cat((v_in, y0+k2/2), dim=1))
            k4 = self.derivative_network(torch.cat((v_in, y0+k3), dim=1))

            output[n, :, :] = y0 + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
            y0 = output[n, :, :]

        return output

    def reset_hidden(self):
        pass

    def detach_hidden(self):
        pass
