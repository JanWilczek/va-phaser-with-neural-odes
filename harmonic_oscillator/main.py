from math import pi
from argparse import ArgumentParser
import torch
from torch import nn
import matplotlib.pyplot as plt
from torchdiffeq import odeint


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
    return ap

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        activation = nn.Identity()
        self.network = nn.Sequential(nn.Linear(2, 100), activation,
                                     nn.Linear(100, 100), activation,
                                     nn.Linear(100, 2))

    def forward(self, t, y):
        return self.network(y)

class HarmonicOscillator():
    """A damped harmonic oscillator with an excitation function
    It is a second-order ODE, which can be rewritten as a system
    of first-order ODEs
    x' = v
    v' = - (k/m) x - (c/m) v + F(t)
    x(0)  = x0
    v(0) = v0
    """
    def __init__(self, m=1, k=1, c=0.1, F=lambda t: 0):
        self.m = m
        self.k = k
        self.c = c
        self.F = F

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
        rhs[1] = - (self.k / self.m) * x - (self.c / self.m) * v + self.F(t)

        return rhs

def get_method(args):
    method_dict = {"odeint": odeint,
                   "odeint_euler": partial(odeint, method='euler'),
                   "ForwardEuler": ForwardEuler(),
                   "trapezoid_rule": trapezoid_rule}
    return method_dict[args.method]

def plot_trajectories(trajectories, time):
    plt.figure()
    trajectories_count = trajectories.shape[1]
    for i in range(trajectories_count):
        plt.subplot(trajectories_count, 1, i+1)
        plt.plot(time, trajectories[:, i, 0])
    plt.savefig('oscillator_trajectories.png', bbox_inches='tight', dpi=300)

def main():
    args = argument_parser().parse_args()
    T = 2 * pi * args.nperiods
    dt = T / args.nsteps
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    oscillator = HarmonicOscillator(m=args.m, k=args.k, c=args.c)

    initial_conditions = [[1, -1], [2, -0.5], [3, 0], [4, 0.5], [5, 1], [-1, 0.5], [-2, 1]]
    trajectories = torch.empty((args.nsteps, len(initial_conditions), 2))   # time step x number of trajectory segments x number of ODEs
    for i, y0 in enumerate(initial_conditions):
        t, trajectories[:, i] = oscillator.synthesize(y0=torch.Tensor(y0), t0=0, t1=T, dt=dt)
    ntest_samples = int(0.2 * args.nsteps)
    # test_samples_indices_start = torch.randint(0, args.nsteps - ntest_samples, (len(initial_conditions),))
    # test_samples_indices_start = torch.ones((len(initial_conditions),), dtype=int) * (args.nsteps // 2)
    test_samples_indices_start = args.nsteps - ntest_samples
    test_samples_indices_end = test_samples_indices_start + ntest_samples
    
    if args.visualize:
        plot_trajectories(trajectories, t)

    network = MLP()
    network.to(device)
    loss_function = nn.MSELoss()
    method = get_method(args)

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    return
    for epoch in tqdm(range(args.epochs)):
        epoch_loss = 0
        # for i, train in enumerate(trainsets):
        optimizer.zero_grad()

        output = method(network, train_y0.to(device), train_t.to(device))
            # output = method(network, train.y0, train.t)

            # loss = loss_function(output, train.true_y)
        loss = loss_function(output, train_target.to(device))

        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        # print(f'Epoch {epoch+1}/{epochs}: Train loss: {epoch_loss/(i+1)}.')
        # print(f'Epoch {epoch+1}/{epochs}: Train loss: {epoch_loss}.')
    
    # Test
    network.to('cpu')
    plt.figure()
    legend = []
    test_loss = 0
    with torch.no_grad():
        for i, test in enumerate(testsets):
            test_output = method(network, test.y0, test.t)
            test_loss += loss_function(test_output, test.true_y)
            plt.subplot(len(testsets), 1, i+1)
            plt.plot(test.t, test.true_y[:,0])
            plt.plot(test.t, test_output[:,0])
            legend += ['Ground truth', 'ODENet output']
    plt.legend(legend)
    i = torch.randint(0,100,1)
    plt.savefig(f'odenet_harmonic_oscillator_{method_name}_{i}.png')
    print(f'{method_name} {i} Test loss: {test_loss}.')


if __name__ == '__main__':
    main()
