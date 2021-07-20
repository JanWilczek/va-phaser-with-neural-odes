import unittest
import torch
import numpy as np
from torchinterp1d import Interp1d


class InterpolationTest(unittest.TestCase):
    def test_middle_same_points(self):
        rows_count = 3
        x = torch.cat((torch.zeros((rows_count, 1)), torch.ones((rows_count,1))), dim=1)
        y = x.clone() + 1.0

        x_new = torch.ones((rows_count, 1)) * 0.5

        y_new = Interp1d()(x, y, x_new)

        np.testing.assert_array_almost_equal(3*x_new, y_new)    

    def test_middle_3_points(self):
        x = torch.cat((torch.zeros((3, 1)), torch.ones((3,1))), dim=1)
        x_new = torch.ones((3, 1)) * 0.5
        y = torch.Tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
        
        y_new = Interp1d()(x, y, x_new)

        y_expected = torch.Tensor([[0.5], [1.0], [1.5]])

        np.testing.assert_array_almost_equal(y_new, y_expected)


if __name__ == '__main__':
    unittest.main()
