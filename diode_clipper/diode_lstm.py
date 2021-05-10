"""Set up an LSTM-RNN architecture, run its training and test on the diode clipper data."""
import torch
import CoreAudioML.networks as networks
import CoreAudioML.training as training
import CoreAudioML.dataset as dataset
from NetworkTraining import NetworkTraining


def create_dataset():
    d = dataset.DatSet(data_dir='data')

    d.create_subset('train', frame_len=22050)
    d.create_subset('validation')
    d.create_subset('ignore')
    d.load_file('diodeclip', set_names=['train', 'validation', 'ignore'], splits=[0.8*0.8, 0.8*0.2, 0.2])

    d.create_subset('test')
    d.load_file('diodeclip', set_names='test')

if __name__ == '__main__':
    training = NetworkTraining()
    
    training.network = networks.SimpleRNN(unit_type="LSTM", hidden_size=8, skip=0)
    training.optimizer = torch.optim.Adam(training.network.parameters(), lr=0.001)
    training.loss = training.ESRLoss()
    
    training.dataset = create_dataset()

    training.epochs = 1
    training.segments_in_a_batch = 40
    training.samples_between_updates = 2048
    training.initialization_length = 1000

    training.run()
