import torch
import torch.nn as nn
import numpy as np
import torch
import math
from scipy.io import wavfile
from pathlib import Path
from architectures import FlexibleStateTrajectoryNetwork

class DataSet:
    def __init__(self, datapath, seg_len):
        samplerate, inp = wavfile.read(datapath + '-input.wav')
        samplerate, tgt = wavfile.read(datapath + '-target.wav')

        self.samplerate = samplerate

        self.ts = 1/self.samplerate
        if seg_len:
            self.data = framify(inp/32768, tgt/32768, seg_len)
        else:
            self.data = framify(inp/32768, tgt/32768, inp.shape[0])


class STN(nn.Module):
    def __init__(self, hidden_size=20, states=2, inputs=1):
        super(STN, self).__init__()
        self.states = states
        self.inputs = inputs
        # Set number of layers  and hidden_size for network layer/s
        # self.lin1 = nn.Linear(states + inputs, hidden_size, bias=False)
        # self.lin2 = nn.Linear(hidden_size, states, bias=False)
        self.densely_connected_layers = nn.Sequential(nn.Linear(states + inputs, hidden_size), nn.Tanh(), nn.Linear(hidden_size, states), nn.Tanh())
        self.state = None
        self.act_dict = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}

    # Define forward pass
    def forward(self, x):
        # fx = torch.tanh(self.lin1(torch.cat((x, self.state), 1)))
        # fx = torch.tanh(self.lin2(fx))
        fx = self.densely_connected_layers(torch.cat((x, self.state), 1))
        self.state = self.state + self.sample_rate*fx
        return self.state

    # Set State
    def set_state(self, state):
        self.state = state

    # Set State
    def set_samp_rate(self, rate):
        self.sample_rate = 1.5*rate/44100


# Loss Functions
class ESRLoss(nn.Module):
    def __init__(self):
        super(ESRLoss, self).__init__()
        self.epsilon = 0.00001

    def forward(self, output, target):
        loss = torch.add(target, -output)
        loss = torch.pow(loss, 2)
        loss = torch.mean(loss)
        energy = torch.mean(torch.pow(target, 2)) + self.epsilon
        loss = torch.div(loss, energy)
        return loss

# converts continuous audio into frames, and creates a torch tensor from them
def framify(inp, target, seg_len):

    # Calculate the number of segments the training data will be split into
    seg_num = math.floor(inp.shape[0] / seg_len)

    # Initialise training and validation set tensor matrices
    dataset = (torch.empty((seg_len, seg_num, 1)),
               torch.empty((seg_len, seg_num, target.shape[1])))

    # Load the audio for the training set
    for i in range(seg_num):
        dataset[0][:, i, 0] = torch.from_numpy(inp[i * seg_len:(i + 1) * seg_len])
        dataset[1][:, i, :] = torch.from_numpy(target[i * seg_len:(i + 1) * seg_len, :])

    return dataset


if __name__ == "__main__":
    datadir = 'diode2_clipper/data/'
    val_data_file = 'validation/diode2clip'
    
    checkpoint_path = 'diode2_clipper/runs/diode2clip/stn/alec_dc2/checkpoint.pth'
    checkpoint = torch.load(checkpoint_path)

    val_dataset = DataSet(datadir + val_data_file, 0)

    alec_stn = STN(hidden_size=20)
    alec_stn.load_state_dict(checkpoint['model_state_dict'])
    alec_stn.set_samp_rate(44100)
    alec_stn.set_state(torch.zeros((1, 2)))
    
    jan_stn = FlexibleStateTrajectoryNetwork([3, 20, 2], nn.Tanh())
    jan_stn.load_state_dict(checkpoint['model_state_dict'])

    loss_fn = ESRLoss()

    with torch.no_grad():
        jan_val_output = jan_stn(val_dataset.data[0])
        jan_val_loss_state1 = loss_fn(jan_val_output[:, :, 0], val_dataset.data[1][:, :, 0])
        jan_val_loss_state2 = loss_fn(jan_val_output[:, :, 1], val_dataset.data[1][:, :, 1])
        print('Jan STN val loss 1: ' + str(jan_val_loss_state1.item()))
        print('Jan STN val loss 2: ' + str(jan_val_loss_state2.item()))
        
        val_out = torch.empty_like(val_dataset.data[1])
        for each in range(val_dataset.data[0].shape[0]):
            val_out[each, :, :] = alec_stn(val_dataset.data[0][each, :, :])
        val_loss_state1 = loss_fn(val_out[:, :, 0], val_dataset.data[1][:, :, 0])
        val_loss_state2 = loss_fn(val_out[:, :, 1], val_dataset.data[1][:, :, 1])
        print('Alec STN val loss 1: ' + str(val_loss_state1.item()))
        print('Alec STN val loss 2: ' + str(val_loss_state2.item()))        
