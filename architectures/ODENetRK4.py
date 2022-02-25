import torch
from .ODENetFE import ScaledODENetFE


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
            u2 = self.derivative_network(t0 + self.dt / 2, y0 + u1 / (2 * self.sampling_rate))
            u3 = self.derivative_network(t0 + self.dt / 2, y0 + u2 / (2 * self.sampling_rate))
            u4 = self.derivative_network(t1, y0 + u3 / self.sampling_rate)

            scaled_y1 = scaled_y0 + 1 / 6 * (u1 + 2 * u2 + 2 * u3 + u4)

            # Assign to output
            y[n] = scaled_y1 / self.sampling_rate

            # Update the loop variables
            scaled_y0 = scaled_y1
            n += 1
        return y
