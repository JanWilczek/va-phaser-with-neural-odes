import torch
from torch import nn
from CoreAudioML.dataset import DataSet
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


def normalized_mse_loss(output, target):
    minimum_value = 1e-5 * torch.ones_like(target)
    loss = torch.mean(torch.div((target - output) ** 2, torch.maximum(target ** 2, minimum_value)))
    return loss

# No teacher forcing as for now
class StateTrajectoryNetwork(nn.Module):
    def __init__(self, is_trained=False):
        super().__init__()
        self.rnn = torch.nn.RNN(input_size=1, hidden_size=8, num_layers=2, nonlinearity='tanh', bias=False)
        self.fcn = torch.nn.Linear(8, 1, bias=False)
        self.hidden = None

    def forward(self, x):
        rnn_out, self.hidden = self.rnn(x, self.hidden)
        out = self.fcn(rnn_out)
        return out + x

    def initialize_state(self, batch_size, state_size):
        self.state = torch.zeros((batch_size, state_size))

    def detach_hidden(self):
        if self.hidden.__class__ == tuple:
            self.hidden = tuple([h.clone().detach() for h in self.hidden])
        else:
            self.hidden = self.hidden.clone().detach()


def train():
    dataset = DataSet(data_dir='./')
    segment_length = 44100
    dataset.create_subset('train', frame_len=segment_length)
    dataset.create_subset('test')
    dataset.load_file('diodeclip', set_names=['train', 'test'], splits=[0.79, 0.21])

    print(f"Training set shape (sequence length, sequence count, features count (1 sample)): {dataset.subsets['train'].data['input'][0].shape}")

    stn = StateTrajectoryNetwork()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} for training.')

    stn.to(device)
    
    optimizer = optim.Adam(stn.parameters(), lr=0.001)
    criterion = normalized_mse_loss

    # Training
    epochs = 100
    segments_in_a_batch = 40
    samples_between_updates = 2048

    loss_history = torch.zeros((epochs,), device=device)
    gradient_norm_history = torch.zeros((epochs,), device=device)

    input_data = dataset.subsets['train'].data['input'][0].to(device)
    target_data = dataset.subsets['train'].data['target'][0].to(device)

    segments_count = input_data.shape[1]
    batch_count = int(np.ceil(segments_count / segments_in_a_batch))

    segments_order = torch.randperm(segments_count)

    for epoch in range(epochs):
        
        for i in range(batch_count-1):
            minibatch_segment_indices = segments_order[i*segments_in_a_batch:(i+1)*segments_in_a_batch]
            input_minibatch = input_data[:, minibatch_segment_indices, :]
            target_minibatch = target_data[:, minibatch_segment_indices, :]
            
            start_id = 0

            while start_id < segment_length:
                end_id = min(start_id + samples_between_updates, segment_length)
                
                optimizer.zero_grad()

                output_minibatch = stn(input_minibatch[start_id:end_id, :, :])

                loss = criterion(output_minibatch, target_minibatch[start_id:end_id, :, :])
                loss.backward()
                optimizer.step()

                stn.detach_hidden()

                start_id += samples_between_updates

        print(f'Epochs {epoch+1}/{epochs}; Loss: {loss.item()}.')

        loss_history[epoch] = loss.item()
        gradient = torch.cat([param.grad.flatten() for param in stn.parameters()])
        gradient_norm_history[epoch] = torch.linalg.norm(gradient)

    print('Finished training.')
    PATH = './diode_clipper_2x8tanhRNN.pth'
    torch.save(stn.state_dict(), PATH)

    plt.figure()
    plt.plot(loss_history.cpu())
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Normalized MSE)')
    plt.savefig('./loss_rnn.png', bbox_inches='tight', dpi=300)

    plt.figure()
    plt.plot(gradient_norm_history.cpu())
    plt.xlabel('Epochs')
    plt.ylabel('Gradient L2 norm')
    plt.savefig('./gradient_norm_rnn.png', bbox_inches='tight', dpi=300)

def main():
    train()


if __name__ == '__main__':
    main()
