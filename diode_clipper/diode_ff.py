import torch
from torch import nn
import numpy as np
import torchaudio
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from audio_utils import print_stats
from models.StateTrajectoryNetwork import StateTrajectoryNetworkFF


def normalized_mse_loss(output, target):
    minimum_value = 1e-5 * torch.ones_like(target)
    loss = torch.mean(torch.div((target - output) ** 2, torch.maximum(target ** 2, minimum_value)))
    return loss

def train():
    # Data pre-processing
    input_filepath = './diodeclip-input.wav'
    target_filepath = './diodeclip-target.wav'
    input_waveform, sample_rate = torchaudio.load(input_filepath)
    target_waveform, sample_rate = torchaudio.load(target_filepath)
    assert input_waveform.shape == target_waveform.shape
    frames_count = input_waveform.shape[1]
    train_frames_count = int(0.8 * frames_count)
    train_input_waveform = input_waveform[0, :train_frames_count]
    train_target_waveform = target_waveform[0, :train_frames_count]

    sequence_length = 2048
    segments_count = train_frames_count // sequence_length
    input_batch = np.zeros((segments_count, sequence_length, 2))
    target_batch = np.zeros((segments_count, sequence_length, 1))
    for i in range(segments_count):
        start_id = i * sequence_length
        end_id = (i + 1) * sequence_length
        input_batch[i, :, 0] = train_input_waveform[start_id:end_id]
        input_batch[i, 1:, 1] = train_target_waveform[start_id:end_id-1]
        target_batch[i, :, 0] = train_target_waveform[start_id:end_id]

    print(f'1 input minibatch shape: {input_batch.shape}')
    print(f'1 target minibatch shape: {target_batch.shape}')
    
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} for training.')

    input_batch = torch.tensor(input_batch, dtype=torch.float, device=device)
    target_batch = torch.tensor(target_batch, dtype=torch.float, device=device)
    
    # Network
    stn = StateTrajectoryNetworkFF()
    stn.to(device)

    # Loss
    optimizer = optim.Adam(stn.parameters(), lr=0.001)
    criterion = normalized_mse_loss
    # criterion = nn.MSELoss()

    # Training
    epochs = 200
    print_loss_every = 200
    segments_in_a_batch = 40
    batch_count = segments_count // segments_in_a_batch

    loss_history = torch.zeros((epochs,), device=device)
    gradient_norm_history = torch.zeros((epochs,), device=device)

    for epoch in range(epochs):
        
        running_loss = 0.0

        for i in range(batch_count):
            input_minibatch = input_batch[i*segments_in_a_batch:(i+1)*segments_in_a_batch, :, :]
            target_minibatch = target_batch[i*segments_in_a_batch:(i+1)*segments_in_a_batch, :, :]
            
            optimizer.zero_grad()

            output_minibatch = stn(input_minibatch)

            loss = criterion(output_minibatch, target_minibatch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % print_loss_every == print_loss_every - 1:
                print('[%d, %5d] loss: %.5f; Running loss: %.5f' % (epoch + 1, i + 1, loss.item(), running_loss/print_loss_every))
                running_loss = 0.
            
        loss_history[epoch] = loss.item()
        gradient = torch.cat([param.grad.flatten() for param in stn.parameters()])
        gradient_norm_history[epoch] = torch.linalg.norm(gradient)

    print('Finished training.')

    PATH = './diode_clipper_2x8tanhFF.pth'
    torch.save(stn.state_dict(), PATH)

    plt.figure()
    plt.plot(loss_history.cpu())
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Normalized MSE)')

    plt.figure()
    plt.plot(gradient_norm_history.cpu())
    plt.xlabel('Epochs')
    plt.ylabel('Gradient L2 norm')

def test():
    # 1 batch, 1-element sequence, 2 variables (input and state, i.e., previous output)
    PATH = './diode_clipper_2x8tanhRNN.pth'
    stn = StateTrajectoryNetworkFF()
    stn.load_state_dict(torch.load(PATH))

    test_input_waveform, _ = torchaudio.load('./test_input.wav')
    test_target_waveform, _ = torchaudio.load('./test_target.wav')
    test_input_waveform.squeeze_(0)
    test_target_waveform.squeeze_(0)
    input_vector = torch.zeros((1, 1, 2), dtype=torch.float)
    output_vector = torch.zeros((1, 1, 1), dtype=torch.float)
    test_output = torch.zeros_like(test_input_waveform.to('cpu'))

    criterion = normalized_mse_loss

    print('Processing test data...')

    with torch.no_grad():
        for i, sample in tqdm(enumerate(test_input_waveform), total=test_input_waveform.shape[0]):
            input_vector[0, 0, 0] = sample
            input_vector[0, 0, 1] = output_vector[0, 0, 0]

            output_vector = stn(input_vector)

            test_output[i] = output_vector[0, 0, 0]

        test_loss = criterion(test_output, test_target_waveform)
        print(f'Test loss: {test_loss:.5f}')

    print_stats(test_output.unsqueeze(0))
    test_output = torch.clamp(test_output, -1., 1.)
    print_stats(test_output.unsqueeze(0))
    torchaudio.save('./test_output.wav', test_output.unsqueeze(0), sample_rate)

def main():
    train()
    test()


if __name__ == '__main__':
    main()
