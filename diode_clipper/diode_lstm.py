"""Set up an LSTM-RNN architecture, run its training and test on the diode clipper data."""
import torch
import CoreAudioML.networks as networks
import CoreAudioML.training as training


if __name__ == '__main__':
    network = networks.SimpleRNN(unit_type="LSTM", hidden_size=8, skip=0)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    loss = training.ESRLoss()