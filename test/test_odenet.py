import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from functools import partial
import unittest
from parameterized import parameterized
from tqdm import tqdm
import torch
from torch import nn
import matplotlib.pyplot as plt
from math import pi
from torchdiffeq import odeint
from models import ODENet, ODENetDerivative
from models.solvers import ForwardEuler, trapezoid_rule


class TestTrapezoidRule(unittest.TestCase):
    def setUp(self):
        self.network = ODENet(ODENetDerivative(), trapezoid_rule, dt=1.0)

    def test_linear(self):
        t = torch.Tensor([0.0, 1.0, 2.0, 3.0])
        y0 = torch.Tensor([1.0])

        y = trapezoid_rule(lambda t, x: 2.0, y0, t)

        self.assertEqual(y[0], 1.0)
        self.assertEqual(y[1], 3.0)
        self.assertEqual(y[2], 5.0)
        self.assertEqual(y[3], 7.0)

    def test_quadratic(self):
        t = torch.Tensor([0.0, 1.0, 2.0, 3.0])
        y0 = torch.Tensor([0.0])

        y = trapezoid_rule(lambda t, x: 2 * t, y0, t)

        plt.figure()
        plt.plot(t, y, t, t ** 2)
        plt.legend(['Trapezoid rule scheme', 'Ground truth'])
        plt.savefig('quadratic_function.png')

    def test_population_growth(self):
        """u' = r * u, u(0) = 100."""
        def f(t, y):
            return 0.1 * y

        t = torch.linspace(0, 20, 40)

        y = trapezoid_rule(f, torch.Tensor([100]), t)

        plt.figure()
        plt.plot(t, y, t, 100 * torch.exp(0.1 * t))
        plt.legend(['Trapezoid rule scheme', 'Ground truth'])
        plt.savefig('population_growth.png')

class TestODENetPopulationGrowth(unittest.TestCase):
    """Verify ODENet's proper implementation on a simple 
    example of population growth.

    Population growth is an IVP described by the following ODE
    dy / dt = r y(t)
    y(0) = y0
    with a known solution
    y(t) = y0 e^(rt)
    """
    class ODENetNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(nn.Linear(1, 5), nn.ReLU(), nn.Linear(5,5), nn.ReLU(), nn.Linear(5, 1))

        def forward(self, t, y):
            return self.network(y)

    @parameterized.expand([
        [odeint, "odeint"],
        [partial(odeint, method='euler'), "odeint_euler"],
        [ForwardEuler(), "ForwardEuler"],
        [trapezoid_rule, "trapezoid_rule"]
    ])
    def test_main(self, method, method_name):
        r = 0.05
        epochs = 100

        network = TestODENetPopulationGrowth.ODENetNetwork()
        train0, test0 = self.synthesize_data(r, y0=10, t0=0, t1=100, dt=1)
        train1, test1 = self.synthesize_data(r, y0=100, t0=0, t1=10, dt=0.5)
        loss_function = nn.MSELoss()

        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

        for epoch in tqdm(range(epochs)):
            epoch_loss = 0
            for i, train in enumerate([train0, train1]):
                optimizer.zero_grad()

                output = method(network, train.y0, train.t)

                # Full function learning (not function derivative learning)
                # output = network(0, train.t.unsqueeze(1))

                loss = loss_function(output, train.true_y)

                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            # print(f'Epoch {epoch+1}/{epochs}: Train loss: {epoch_loss}.')
        
        testsets = [test0, test1]
        plt.figure()
        legend = []
        test_loss = 0
        with torch.no_grad():
            for i, test in enumerate(testsets):
                test_output = method(network, test.y0, test.t)

                # True derivative function
                # test_output = odeint(partial(self.f, r=r), test.y0, test.t)

                # Full function learning (not function derivative learning)
                # test_output = network(0, test.t.unsqueeze(1))

                test_loss += loss_function(test_output, test.true_y)
                plt.subplot(len(testsets), 1, i+1)
                plt.plot(test.t, test.true_y)
                plt.plot(test.t, test_output)
                legend += ['Ground truth', 'ODENet output']
        plt.legend(legend)
        plt.savefig(f'odenet_population_growth_{method_name}.png')
        print(f'{method_name} Test loss: {test_loss}.')

    def synthesize_data(self, r, y0, t0, t1, dt):
        class Dataset:
            def __init__(self, t, true_y):
                self.y0 = true_y[0]
                self.t = t
                self.true_y = true_y

        t = torch.arange(t0, t1, dt, dtype=torch.float)
        true_y = y0 * torch.exp(r * t).unsqueeze(1)

        plt.figure()
        plt.plot(t, true_y)
        plt.savefig('population_growth_full.png')

        train_set_size = int(t.shape[0] * 0.8)
        test_set_size = t.shape[0] - train_set_size

        train_t, test_t = t.split([train_set_size, test_set_size])
        train_true_y, test_true_y = true_y.split([train_set_size, test_set_size])
        train_set = Dataset(train_t, train_true_y)
        test_set = Dataset(test_t, test_true_y)

        return train_set, test_set

    def f(self, t, y, r):
        """True derivative

        Parameters
        ----------
        t : numeric
            time
        y : numeric
            y's value at t
        r : numeric
            rate of growth

        Returns
        -------
        numeric
            value of the derivative at t
        """
        return r * y

class TestODENetHarmonicOscillator(unittest.TestCase):
    """A damped harmonic oscillator with an excitation function
    It is a second-order ODE, which can be rewritten as a system
    of first-order ODEs
    x' = v
    v' = - (k/m) x - (c/m) v + F(t)
    x(0)  = x0
    v(0) = v0
    """
    m = 1
    k = 1
    c = 0.1
    F = lambda t: 0

    class ODENetNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(nn.Linear(2, 100),
                                         nn.Linear(100, 100),
                                         nn.Linear(100, 2))

        def forward(self, t, y):
            return self.network(y)

    @parameterized.expand([
        [odeint, "odeint"],
        [partial(odeint, method='euler'), "odeint_euler"],
        [ForwardEuler(), "ForwardEuler"],
        [trapezoid_rule, "trapezoid_rule"]
    ])
    def test_main(self, method, method_name):
        epochs = 100
        T = 16 * pi
        dt = T / 5000

        trainsets = []
        testsets = []
        for y0 in [[1, -1], [2, -0.5], [3, 0], [4, 0.5], [5, 1]]:
            train, test = self.synthesize_data(y0=torch.Tensor(y0), t0=0, t1=T, dt=dt)
            trainsets.append(train)
            testsets.append(test)

        network = TestODENetHarmonicOscillator.ODENetNetwork()
        loss_function = nn.MSELoss()

        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

        for epoch in tqdm(range(epochs)):
            epoch_loss = 0
            for i, train in enumerate(trainsets):
                optimizer.zero_grad()

                output = method(network, train.y0, train.t)

                loss = loss_function(output, train.true_y)

                loss.backward()
                optimizer.step()
            
                epoch_loss += loss.item()
            # print(f'Epoch {epoch+1}/{epochs}: Train loss: {epoch_loss/(i+1)}.')
        
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
        plt.savefig(f'odenet_harmonic_oscillator_{method_name}.png')
        print(f'{method_name} Test loss: {test_loss}.')

    def synthesize_data(self, y0, t0, t1, dt):
        class Dataset:
            def __init__(self, t, true_y):
                self.y0 = true_y[0, :]
                self.t = t
                self.true_y = true_y

        t = torch.arange(t0, t1, dt, dtype=torch.float)
        true_y = torch.zeros((t.shape[0], y0.shape[0]), dtype=torch.float)
        with torch.no_grad():
            true_y[0, :] = y0
            for n in range(true_y.shape[0]-1):
                fn = self.f(t[n], true_y[n,:])
                # Update velocity
                true_y[n+1, 1] = true_y[n,1] + dt * fn[1]
                # Update displacement
                true_y[n+1, 0] = true_y[n,0] + dt * true_y[n+1,1]

        plt.figure()
        plt.plot(t, true_y[:,0])
        plt.savefig(f'damped_harmonic_oscillation_{y0}.png')

        train_set_size = int(t.shape[0] * 0.8)
        test_set_size = t.shape[0] - train_set_size

        train_t, test_t = t.split([train_set_size, test_set_size])
        train_true_y, test_true_y = true_y.split([train_set_size, test_set_size])
        train_set = Dataset(train_t, train_true_y)
        test_set = Dataset(test_t, test_true_y)

        return train_set, test_set

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
        rhs[1] = - (self.k / self.m) * x - (self.c / self.m) * v + TestODENetHarmonicOscillator.F(t)

        return rhs


if __name__ == '__main__':
    unittest.main()
