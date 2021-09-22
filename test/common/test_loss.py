import unittest
import torch
from common.loss import get_loss_function

class LogSpectralDistanceTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.loss = get_loss_function('log_spectral_distance')
    
    def test_same(self):
        signal = torch.randn(2048, 256, 1)
        self.assertAlmostEqual(self.loss(signal, signal), 0.)

if __name__ == '__main__':
    unittest.main()
