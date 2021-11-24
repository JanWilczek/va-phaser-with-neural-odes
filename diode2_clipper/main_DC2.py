import torch
import torch.nn as nn
import numpy as np
import torch
import math
from scipy.io import wavfile
from pathlib import Path

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

    batch_size = 25
    epochs = 100
    up_fr = 100

    datadir = 'diode2_clipper/data/'
    datapath = 'train/diode2clip'
    val_data_file = 'validation/diode2clip'

    training_dataset = DataSet(datadir + datapath, 22050)
    val_dataset = DataSet(datadir + val_data_file, 0)

    network = STN(hidden_size=20)
    network.set_samp_rate(44100)

    optimizer = torch.optim.Adam(network.parameters(), lr=0.0005)

    loss_fn = ESRLoss()

    prin_it = 5

    for epochs in range(epochs):
        checkpoint_dict = {
            'epoch': epochs,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }

        torch.save(checkpoint_dict, Path('diode2_clipper', 'runs', 'diode2clip','stn','alec_dc2', 'checkpoint.pth'))
        
        shuffle = torch.randperm(training_dataset.data[0].shape[1])

        epoch_loss = 0
        running_loss = 0
        for each in range(int(training_dataset.data[0].shape[1] / batch_size)):
            batch = training_dataset.data[0][:, shuffle[each * batch_size:(each + 1) * batch_size], :]
            target = training_dataset.data[1][:, shuffle[each * batch_size:(each + 1) * batch_size], :]

            batch_loss = 0
            for chunk in range(int((batch.shape[0]-1)/up_fr)):

                network.set_state(target[chunk*up_fr, :, :])
                output = torch.empty((up_fr, target.shape[1], target.shape[2]))
                for sample in range(up_fr):
                    #output[sample, :, :] = network(batch[1 + chunk*up_fr + sample, :, :])
                    output[sample, :, :] = network(batch[chunk * up_fr + sample, :, :])


                #loss = loss_fn(output, target[1 + chunk*up_fr:1 + (chunk+1)*up_fr, :, :])
                loss = loss_fn(output, target[chunk * up_fr:1 + (chunk) * up_fr, :, :])
                loss.backward()
                optimizer.step()
                network.zero_grad()
                network.state = network.state.detach()
                batch_loss += loss.item()

            epoch_loss += batch_loss/(chunk+1)
            print('mini_batch loss: ' + str(batch_loss/(chunk+1)))

            if (each + 1) % prin_it == 0:
                print('Epoch ' + str(100*(each + 1)/(int(training_dataset.data[0].shape[1] / batch_size))) + '% complete')

        print('Epoch Loss: ' + str(epoch_loss / (each + 1)))

        
        
        if epochs % 5 == 0:
            with torch.no_grad():
                val_out = torch.empty_like(val_dataset.data[1])
                network.set_state(torch.zeros((1, 2)))
                for each in range(val_dataset.data[0].shape[0]):
                    val_out[each, :, :] = network(val_dataset.data[0][each, :, :])
                val_loss_state1 = loss_fn(val_out[:, :, 0], val_dataset.data[1][:, :, 0])
                val_loss_state2 = loss_fn(val_out[:, :, 1], val_dataset.data[1][:, :, 1])
                print('Epoch' + str(epochs + 1) + ', val loss 1: ' + str(val_loss_state1.item()))
                print('Epoch' + str(epochs + 1) + ', val loss 2: ' + str(val_loss_state2.item()))
        
        
