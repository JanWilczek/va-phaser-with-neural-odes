import torch
import torch.nn as nn
from .ODENetFE import ScaledODENetFE


class DerivativeMLPRK4(nn.Module):
    def __init__(self, activation, excitation_size=1, output_size=1, hidden_size=100):
        super().__init__()
        self.excitation_size = excitation_size
        self.output_size = output_size
        self.densely_connected_layers = nn.Sequential(
            nn.Linear(self.input_size, hidden_size), activation,
            nn.Linear(hidden_size, hidden_size), activation,
            nn.Linear(hidden_size, self.output_size))

    def forward(self, t: torch.Tensor, y: torch.Tensor):
        FEATURE_DIMENSION = 1

        if t.is_floating_point():
            floor_t = t.int()
            ceil_t = t.int() + 1
            excitation = 0.5 * (self.excitation[floor_t] + self.excitation[ceil_t])
        else:
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


class ScaledODENetRK4(ScaledODENetFE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def ode_solve(self):
        # After: https://mathworld.wolfram.com/Runge-KuttaMethod.html
        y = torch.empty((self.time.shape[0], *self.state.shape), dtype=self.state.dtype, device=self.device)
        y0 = y[0] = self.state
        scaled_y0 = self.sampling_rate * y0
        n = 1
        for t0, t1 in zip(self.time[:-1], self.time[1:]):
            # u1 = k1 * sampling_rate; u2 = k2 * sampling_rate, etc.
            u1 = self.derivative_network(t0, y0)
            # Time step is assumed to be equal to 1 for the derivative network
            u2 = self.derivative_network(t0 + 0.5, y0 + u1 / (2 * self.sampling_rate))
            u3 = self.derivative_network(t0 + 0.5, y0 + u2 / (2 * self.sampling_rate))
            u4 = self.derivative_network(t1, y0 + u3 / self.sampling_rate)

            scaled_y1 = scaled_y0 + 1 / 6 * (u1 + 2 * u2 + 2 * u3 + u4)

            # Assign to output
            y[n] = scaled_y1 / self.sampling_rate

            # Update the loop variables
            scaled_y0 = scaled_y1
            n += 1
        return y
