import torch
from torch import nn


class ForwardEuler(nn.Module):
    def __init__(self):
        super().__init__()

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
        ndarray
            y values at points specified in t
        """
        y = torch.zeros((t.shape[0], y0.shape[0]), device=t.device)
        y[0, :] = y0

        for n in range(y.shape[0] - 1):
            y[n + 1, :] = y[n, :] + (t[n+1] - t[n]) * f(t[n], y[n, :], *args)
            # y[n + 1, :] = y[n, :] + f(t[n], y[n, :], *args) # STN equivalent, for debugging purposes only

        return y
