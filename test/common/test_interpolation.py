import unittest
import torch
import numpy as np
from architectures import ExcitationSecondsLinearInterpolation


class ExcitationSecondsLinearInterpolationTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.interpolator = ExcitationSecondsLinearInterpolation()
    
    def test_middle_same_points(self):
        minibatch_size = 3
        x = torch.cat((torch.zeros((1, minibatch_size)), torch.ones((1, minibatch_size))), dim=0).unsqueeze(2)
        self.assertEqual(x.shape, (2, minibatch_size, 1))

        # Interpolation output does not contain the time dimension (dimension 0 in x)
        y_expected = 0.5 * torch.ones((minibatch_size, 1))
        
        time = torch.Tensor([0.0, 1.0])

        self.interpolator.set_excitation_data(time, x)

        y_output = self.interpolator(0.5)

        np.testing.assert_array_almost_equal(y_expected, y_output)    

    def test_middle_3_points(self):
        time = torch.Tensor([0.0, 1.0])
        y = torch.Tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]).transpose(0, 1)
        
        self.interpolator.set_excitation_data(time, y)
        y_output = self.interpolator(0.5)

        y_expected = torch.Tensor([0.5, 1.0, 1.5])

        np.testing.assert_array_almost_equal(y_output, y_expected)


if __name__ == '__main__':
    unittest.main()
