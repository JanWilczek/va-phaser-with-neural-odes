import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from functools import partial
import unittest
import torch
from torch import nn
import matplotlib.pyplot as plt
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
            self.network = nn.Sequential(nn.Linear(1, 5), nn.ReLU(), nn.Linear(5, 1), nn.ReLU())

        def forward(self, t, y):
            return self.network(y)

    def test_main(self):
        r = 0.05
        y0 = 10
        network = TestODENetPopulationGrowth.ODENetNetwork()
        train0, test0 = self.synthesize_data(r, y0, 0, 100, 1)
        epochs = 100
        loss_function = nn.MSELoss()

        optimizer = torch.optim.Adam(network.parameters())

        for epoch in range(epochs):
            epoch_loss = 0
            for i, train in enumerate([train0]):
                optimizer.zero_grad()

                output = odeint(network, train.y0, train.t, method='euler')
                # output = ForwardEuler()(network, train.y0, train.t)
                loss = loss_function(output, train.true_y)

                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            print(f'Epoch {epoch+1}/{epochs}: Train loss: {epoch_loss}.')
        
        plt.figure()
        legend = []
        test_loss = 0
        with torch.no_grad():
            for i, test in enumerate([test0]):
                # test_output = odeint(partial(self.f, r=r), test.y0, test.t) # works great
                test_output = odeint(network, test.y0, test.t, method='euler')
                # test_output = ForwardEuler()(network, test.y0, test.t)
                test_loss += loss_function(test_output, test.true_y)
                plt.plot(test.t, test.true_y)
                plt.plot(test.t, test_output)
                legend += ['Ground truth', 'ODENet output']
        plt.legend(legend)
        plt.savefig('odenet_population_growth.png')
        print(f'Test loss: {test_loss}.')


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

        Returns
        -------
        numeric
            value of the derivative at t
        """
        return r * y


if __name__ == '__main__':
    unittest.main()
