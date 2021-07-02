import torch
from torch import nn


class ForwardEuler:
    def __init__(self):
        super().__init__()

    def __call__(self, f, y0, t, args=[], **kwargs):
        return self.forward(f, y0, t, args=[], **kwargs)

    def forward(self, f, y0, t, args=[], **kwargs):
        """Euler scheme of solving an ordinary differential 
        equation (ODE) numerically.

        Parameters
        ----------
        f : callable
            right-hand side of the ODE
            dy / dt = f(t, y),
            where t and y are a scalar and a (N,) torch.Tensor respectively
        y0 : torch.Tensor
            initial value of the function to be calculated
            shape (N,) where N is the number of dimensions
        t : torch.Tensor
            time points to evaluate y at
            t[0] corresponds to y0
            shape (K,) where K is the number of time points
        args: additional arguments to be passed to f

        Returns
        -------
        torch.Tensor
            y values at points specified in t
        """
        y = torch.empty((t.shape[0], *y0.shape), dtype=y0.dtype, device=t.device)
        y[0, ...] = y0

        n = 1
        for t0, t1 in zip(t[:-1], t[1:]):
            y1 = y0 + (t1 - t0) * f(t0, y0, *args)
            y[n, ...] = y1
            n += 1
            y0 = y1

        return y
