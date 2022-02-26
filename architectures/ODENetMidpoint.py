import torch
from .ODENetFE import ScaledODENetFE


class ScaledODENetMidpoint(ScaledODENetFE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def ode_solve(self):
        # Explicit midpoint method
        # After: https://en.wikipedia.org/wiki/Midpoint_method
        y = torch.empty((self.time.shape[0], *self.state.shape), dtype=self.state.dtype, device=self.device)
        y0 = y[0] = self.state
        scaled_y0 = self.sampling_rate * y0
        n = 1
        for t0, t1 in zip(self.time[:-1], self.time[1:]):
            # Time step is assumed to be equal to 1 for the derivative network
            u1 = 0.5 * self.derivative_network(t0, y0)
            u2 = self.derivative_network(t0 + 0.5, y0 + u1 / self.sampling_rate)

            scaled_y1 = scaled_y0 + u2

            # Assign to output
            y1 = scaled_y1 / self.sampling_rate
            y[n] = y1

            # Update the loop variables
            y0 = y1
            scaled_y0 = scaled_y1
            n += 1
        return y
