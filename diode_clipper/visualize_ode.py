import argparse
import json
import torch
from common import NetworkTraining
from architectures import get_diode_clipper_architecture


def main():
    # Load model
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_path', help='Path to the directory with the run to analyze model from.')
    args = ap.parse_args()
    session = NetworkTraining()
    session.run_directory = args.run_path
    args = json.load(session.run_directory / 'args.json')
    dt = 1 / 44100
    session.network = get_diode_clipper_architecture(args, dt)
    session.load_checkpoint(best_validation=True)
    network = session.network

    torch.no_grad()
    # arange input and state space
    step = 1e-2
    input = torch.arange(-1, 1, step)
    state = torch.arange(-1, 1, step)
    time = dt * torch.arange(0, state.shape[0])

    input2D, state2D  = torch.meshgrid(input, state)

    # For each point calculate the derivative
    derivative_magnitude = torch.zeros(input.shape[0], state.shape[0])
    network.set_excitation_data(time, input)
    for i, t in enumerate(time):
        derivative_magnitude[i, :] = torch.abs(network(t, state))

    # Plot input vs state and display the magnitude of the learned derivative
    box = (-1, 1, -1, 1)
    plt.figure()
    plt.imshow(derivative_magnitude,
                extent=box, origin='lower')
    plt.savefig(session.run_directory / 'derivative.png', bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    main()
