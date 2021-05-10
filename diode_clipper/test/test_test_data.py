from pathlib import Path
import unittest
import torch
import torchaudio


class TestTestData(unittest.TestCase):
    def setUp(self):
        self.datapath = Path('diode_clipper')
    
    def test_test_data(self):
        test_output_signal, test_output_sample_rate = torchaudio.load(self.datapath / 'test_output.wav')
        test_input_signal, test_input_sample_rate = torchaudio.load(self.datapath / 'test_input.wav')
        test_target_signal, test_target_sample_rate = torchaudio.load(self.datapath / 'test_target.wav')

        self.assertEqual(test_output_sample_rate, test_input_sample_rate)
        self.assertEqual(test_input_sample_rate, test_target_sample_rate)

        self.assertEqual(test_output_signal.shape, test_input_signal.shape)
        self.assertEqual(test_target_signal.shape, test_input_signal.shape)


if __name__ == '__main__':
    unittest.main()
