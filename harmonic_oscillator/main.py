#!/usr/bin/env python3
"""Sample call
python harmonic_oscillator/main.py --visualize --epochs 100 -m 1 -k 1 -c 0.1 --nsteps 5000 --nperiods 8 --method forward_euler --excitation 1.8 0.8 --name TimeInSamples --interpolation linear --duffing 0.8
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math import pi
from functools import partial
from argparse import ArgumentParser
import torch
from torch import nn
import matplotlib.pyplot as plt
from torchdiffeq import odeint, odeint_adjoint
from solvers import ForwardEuler, trapezoid_rule
from excitation import ExcitationSeconds, ExcitationSamples, ExcitationSecondsInterpolation0, ExcitationSecondsInterpolation1
from common import setup_pyplot_for_latex, save_tikz


def argument_parser():
    ap = ArgumentParser()
    ap.add_argument('--method', default='forward_euler')
    ap.add_argument('--epochs', type=int)
    ap.add_argument('-m', default=1.0, type=float)
    ap.add_argument('-k', default=1.0, type=float)
    ap.add_argument('-c', default=0.3, type=float)
    ap.add_argument('--nsteps', default=5000, type=int)
    ap.add_argument('--nperiods', default=8, type=int)
    ap.add_argument('--visualize', action='store_true')
    ap.add_argument('--use_samples', action='store_true')
    ap.add_argument('--interpolation', choices=['exact', 'zero', 'linear'])
    ap.add_argument('--segment_size', default=100, type=int)
    ap.add_argument('--name', default="")
    ap.add_argument('--excitation', type=float, default=[0.0, 0.0], nargs=2)
    ap.add_argument('--duffing', type=float, default=0.0)
    return ap

class HarmonicOscillator():
    """A damped harmonic oscillator with an excitation function
    It is a second-order ODE, which can be rewritten as a system
    of first-order ODEs
    x' = v
    v' = - (k/m) x - (c/m) v + F(t) - e x^3
    x(0) = x0
    v(0) = v0
    
    e is the epsilon coefficient of the Duffing equation (an anharmonic oscillator) to make the system of ODEs nonlinear.
    """
    def __init__(self, m=1, k=1, c=0.1, F=lambda t: 0, duffing=0.0):
        self.m = m
        self.k = k
        self.c = c
        self.F = F
        self.duffing = duffing

    def synthesize(self, y0, t0, t1, dt):
        t = torch.arange(t0, t1, dt, dtype=torch.float)
        y = torch.zeros((t.shape[0], *y0.shape), dtype=torch.float)
        with torch.no_grad():
            y[0, ...] = y0
            for n in range(t.shape[0] - 1):
                fn = self.f(t[n], y[n,:])
                # Update velocity
                y[n+1, 1] = y[n, 1] + dt * fn[1]
                # Update displacement
                y[n+1, 0] = y[n,0] + dt * y[n+1,1]

        return t, y

    def f(self, t, y):
        """True right-hand side of the ODE

        Parameters
        ----------
        t : numeric
            time
        y : numeric
            y's value at t

        Returns
        -------
        numeric
            value of the derivative at t
        """
        x = y[0]
        v = y[1]

        rhs = torch.zeros_like(y)

        rhs[0] = v
        rhs[1] = - (self.k / self.m) * x - (self.c / self.m) * v + self.F(t) - self.duffing * x ** 3

        return rhs

class MLP(nn.Module):
    def __init__(self, excitation):
        super().__init__()
        self.excitation = excitation
        # activation = nn.Identity()
        activation = nn.ReLU()
        self.network = nn.Sequential(nn.Linear(3, 100), activation,
                                     nn.Linear(100, 100), activation,
                                     nn.Linear(100, 2))

    def forward(self, t, y):
        y_with_excitation = torch.cat((y, torch.tile(self.excitation(t), (y.shape[0], 1))), dim=1)
        return self.network(y_with_excitation)
        
def get_method(args):
    method_dict = {"odeint": odeint,
                   "odeint_euler": partial(odeint, method='euler'),
                   "implicit_adams": partial(odeint, method='implicit_adams'),
                   "forward_euler": ForwardEuler(),
                   "trapezoid_rule": trapezoid_rule}
    return method_dict[args.method]

def get_excitation(args, dt):
    amplitude, frequency = args.excitation
    excitation_dict = {'exact': ExcitationSeconds(amplitude, frequency),
                       'zero': ExcitationSecondsInterpolation0(amplitude, frequency, dt),
                       'linear': ExcitationSecondsInterpolation1(amplitude, frequency, dt)}
    return excitation_dict[args.interpolation]

def plot_trajectories(trajectories, time, estimated_trajectories=None):
    # setup_pyplot_for_latex()
    trajectories_count = trajectories.shape[1]
    fig = plt.figure()
    gs = fig.add_gridspec(trajectories_count, ncols=1, hspace=0)
    axes = gs.subplots(sharex=True, sharey=True)
    for i in range(trajectories_count):
        axes[i].plot(time, trajectories[:, i, 0])
        if estimated_trajectories is not None:
            axes[i].plot(time, estimated_trajectories[:, i, 0])
        axes[i].label_outer()   
    
    if estimated_trajectories is not None:
        legend = ["Ground truth", "ODENet"]
        fig.legend(legend)


def main():
    args = argument_parser().parse_args()
    T = 2 * pi * args.nperiods
    dt = T / args.nsteps
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    excitation_seconds = get_excitation(args, dt)
    oscillator = HarmonicOscillator(m=args.m, k=args.k, c=args.c, F=excitation_seconds, duffing=args.duffing)

    # initial_conditions = [[1, -1], [2, -0.5], [3, 0], [4, 0.5], [5, 1], [-1, 0.5], [-2, 1]]
    initial_conditions = [[0, 1], [0, 0]]
    trajectories = torch.empty((args.nsteps, len(initial_conditions), 2))   # time step x number of trajectory segments x number of ODEs
    for i, y0 in enumerate(initial_conditions):
        t_seconds, trajectories[:, i] = oscillator.synthesize(y0=torch.Tensor(y0), t0=0, t1=T, dt=dt)
    t = t_seconds

    mlp_excitation = excitation_seconds

    if args.use_samples:
        t_samples = torch.arange(0, args.nsteps, dtype=torch.float)
        amplitude, frequency = args.excitation
        excitation_samples = ExcitationSamples(amplitude, frequency, dt)

        excitation_samples_values = torch.Tensor([excitation_samples(n) for n in t_samples]).unsqueeze(1).unsqueeze(2)
        excitation_seconds_values = torch.Tensor([excitation_seconds(t_sec) for t_sec in t]).unsqueeze(1).unsqueeze(2)
        plot_trajectories(excitation_samples_values, t_samples)
        plt.savefig('excitation_samples_values.png', bbox_inches='tight', dpi=300)
        plot_trajectories(excitation_samples_values, t)
        plt.savefig('excitation_seconds_values.png', bbox_inches='tight', dpi=300)
        
        t = t_samples # Use sample-based indexing from now on
        mlp_excitation = excitation_samples

    ntest_samples = int(0.2 * args.nsteps)
    test_samples_indices_start = args.nsteps - ntest_samples
    test_samples_indices_end = test_samples_indices_start + ntest_samples
    
    if args.visualize:
        plot_trajectories(trajectories, t)
        plt.savefig('oscillator_trajectories.png', bbox_inches='tight', dpi=300)
        save_tikz('oscillator_trajectories')

    network = MLP(mlp_excitation)
    network.to(device)
    loss_function = nn.MSELoss()
    method = get_method(args)

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    try:
        for epoch in range(args.epochs):
            epoch_loss = 0

            for i in range(test_samples_indices_start // args.segment_size):
                optimizer.zero_grad()

                trajectories_segment = trajectories[i*args.segment_size:(i+1)*args.segment_size].to(device)
                t_segment = t[i*args.segment_size:(i+1)*args.segment_size].to(device)

                output = method(network, trajectories_segment[0], t_segment)

                loss = loss_function(output, trajectories_segment)

                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            print(f'Epoch {epoch+1}/{args.epochs}: Train loss: {epoch_loss/(i+1)}.')
    except KeyboardInterrupt:
        pass
    except Exception:
        raise
    
    # Test
    network.to('cpu')
    with torch.no_grad():
        test_trajectories = trajectories[test_samples_indices_start:]
        test_t = t[test_samples_indices_start:]
        test_output = method(network, test_trajectories[0], test_t) # ground truth initialization
        # test_output = method(network, torch.zeros_like(test_trajectories[0]), test_t) # all zeros initialization
        test_loss = loss_function(test_output, test_trajectories)

    i = torch.randint(0,1000,(1,))[0]
    result = f'{args.method} {i} {args.name} Test loss: {test_loss}.'
    print(result)

    if args.visualize:
        plot_trajectories(test_trajectories, test_t, test_output)
        plt.title(result)
        plt.savefig(f'odenet_harmonic_oscillator_{args.method}_{i}_{args.name}.png')


if __name__ == '__main__':
    main()
