from argparse import ArgumentParser
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
from common import create_dataset
from solvers import ForwardEuler


def argument_parser():
    ap = ArgumentParser()
    ap.add_argument('datasets_folder', help='path to the folder with \"train\", \"test\", \
                    and \"validation\" folders, e.g., \"phaser/data\"')
    ap.add_argument('dataset_name', help='name of the dataset to extract the derivative value from, e.g. \"FameSweetToneOffNoFb\"')
    return ap

def main():
    args = argument_parser().parse_args()
    d = create_dataset(args.datasets_folder, args.dataset_name)
    sampling_rate = d.subsets["validation"].fs
    dt = 1 / sampling_rate
    validation_set = d.subsets["validation"].data['target'][0]
    derivative_network_target = sampling_rate * torch.diff(validation_set[:, 0, 0], dim=0, append=torch.zeros((1,)))
    data = derivative_network_target.squeeze().cpu().detach().numpy()
    path = f'{args.dataset_name}-derivative-network-target.wav'
    wavfile.write(path, sampling_rate, data)
    time = torch.arange(0, validation_set.shape[0] / sampling_rate, dt, dtype=torch.float64)

    reconstruction = ForwardEuler()(lambda t, y0: derivative_network_target[int(np.round(t * sampling_rate))], validation_set[0], time, dt=dt)

    np.testing.assert_array_almost_equal(reconstruction.detach().cpu().numpy(), validation_set.detach().cpu().numpy())


if __name__=='__main__':
    """Extract the derivative value that the derivative network should learn in the forward Euler scheme and store in a file. For the derivative value, the validation set is used."""
    main()
