import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import torch
import matplotlib.pyplot as plt
from models import ODENet, ODENetDerivative
from models.solvers import forward_euler


class TestTrapezoidRule(unittest.TestCase):
    def setUp(self):
        self.network = ODENet(ODENetDerivative(), forward_euler, dt=1.0)

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

if __name__ == '__main__':
    unittest.main()
