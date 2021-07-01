import torch


def trapezoid_rule(f, y0, t, args=[], rtol=1e-4, **kwargs):
    """Trapezoid rule scheme of solving an ordinary differential 
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
    args: list
        additional arguments to be passed to f
    implicit_iters: int
        number of refinements of the next point estimate

    Returns
    -------
    ndarray
        y values at points specified in t
    """
    y = torch.zeros((t.shape[0], y0.shape[0]), device=t.device)
    y[0, :] = y0

    n = 1
    for t0, t1 in zip(t[:-1], t[1:]):
        dt = t1 - t0
        fn = f(t0, y0, *args)
        y1 = y0 + dt * fn # forward Euler
        relative_error = 2 * rtol
        while relative_error > rtol:
            y1_new = y0 + 0.5 * dt * (fn + f(t1, y1.clone(), *args))
            relative_error = (y1_new - y1) / (y1 + 1e-6)
            y1 = y1_new
        y[n, :] = y1
        y0 = y1
        n += 1

    return y
