from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
from common import create_dataset
from solvers import ForwardEuler


def main():
    dataset_name = 'FameSweetToneOffNoFb'
    d = create_dataset('phaser/data', dataset_name)
    sampling_rate = d.subsets["validation"].fs
    dt = 1 / sampling_rate
    validation_set = d.subsets["validation"].data['target'][0]
    derivative_network_target = sampling_rate * torch.diff(validation_set[:, 0, 0], dim=0, append=torch.zeros((1,)))
    data = derivative_network_target.squeeze().cpu().detach().numpy()
    path = f'{dataset_name}-derivative-network-target.wav'
    wavfile.write(path, sampling_rate, data)
    time = torch.arange(0, validation_set.shape[0] / sampling_rate, dt, dtype=torch.float64)

    reconstruction = ForwardEuler()(lambda t, y0: derivative_network_target[int(np.round(t * sampling_rate))], validation_set[0], time, dt=dt)

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(validation_set.squeeze())
    plt.subplot(3, 1, 2)
    plt.plot(reconstruction.squeeze())
    plt.subplot(3, 1, 3)
    plt.plot(derivative_network_target.squeeze())
    plt.savefig('derivative_network_target.png', bbox_inches='tight', dpi=300)
    
    np.testing.assert_array_almost_equal(reconstruction.detach().cpu().numpy(), validation_set.detach().cpu().numpy())


if __name__=='__main__':
    main()
