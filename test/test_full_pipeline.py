import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from pathlib import Path
import torch
import torchaudio
import CoreAudioML.training as training
import CoreAudioML.networks as networks
from NetworkTraining import NetworkTraining, create_dataset
from models.StateTrajectoryNetwork import StateTrajectoryNetworkFF


class TestFullPipeline(unittest.TestCase):
    def test_minimal_pipeline(self):
        session = NetworkTraining()

        session.run_directory = Path('diode_clipper', 'runs', 'test')
        
        session.device = 'cpu'

        session.network = networks.SimpleRNN(unit_type="LSTM", hidden_size=2, skip=0)
        session.optimizer = torch.optim.Adam(session.network.parameters(), lr=0.001)
        session.loss = training.ESRLoss()
        
        session.dataset = create_dataset()

        session.epochs = 2
        session.segments_in_a_batch = 40
        session.samples_between_updates = 2048
        session.initialization_length = 1000

        session.run()

        session.load_checkpoint()

        test_output, test_loss = session.test()

        self.assertEqual(test_output.shape, session.input_data('test').shape)


if __name__ == '__main__':
    unittest.main()
