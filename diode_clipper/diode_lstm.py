"""Set up an LSTM-RNN architecture, run its training and test on the diode clipper data."""
from pathlib import Path
import torch
import CoreAudioML.networks as networks
import CoreAudioML.training as training
import CoreAudioML.dataset as dataset
from NetworkTraining import NetworkTraining


def create_dataset():
    d = dataset.DataSet(data_dir=str(Path('diode_clipper', 'data').resolve()))

    d.create_subset('train', frame_len=22050)
    d.create_subset('validation')
    d.create_subset('ignore')
    d.load_file('diodeclip', set_names=['train', 'validation', 'ignore'], splits=[0.8*0.8, 0.8*0.2, (1.0 - 0.8*0.8 - 0.8*0.2)])

    d.create_subset('test')
    d.load_file('diodeclip', set_names='test')

    return d

if __name__ == '__main__':
    session = NetworkTraining()
    
    session.network = networks.SimpleRNN(unit_type="LSTM", hidden_size=8, skip=0)
    session.optimizer = torch.optim.Adam(session.network.parameters(), lr=0.001)
    session.loss = training.ESRLoss()
    
    session.dataset = create_dataset()

    session.epochs = 1
    session.segments_in_a_batch = 40
    session.samples_between_updates = 2048
    session.initialization_length = 1000
    session.model_store_path = Path('diode_clipper', 'models', 'lstm_8.pth').resolve()

    session.run()
