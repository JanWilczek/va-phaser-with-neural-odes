import argparse
import json
import matplotlib.pyplot as plt
import torch
from common import NetworkTraining, argument_parser
from architectures import get_diode_clipper_architecture


def main():
    # Load model
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_path', help='Path to the directory with the run to analyze model from.')
    session = NetworkTraining()
    session.run_directory = ap.parse_args().run_path
    with open(session.run_directory / 'args.json', 'r') as f:
        args_dict = json.load(f)
    args = argument_parser().parse_args(['--method', args_dict['method'],
                                        '-eps', str(args_dict['epochs']),
                                        '-bs', str(args_dict['batch_size']),
                                        '--dataset_name', 'diodeclip'])
    args.__dict__.update(args_dict)
    dt = 1 / 44100
    session.network = get_diode_clipper_architecture(args, dt)
    session.optimizer = torch.optim.Adam(
        session.network.parameters(),
        lr=args.learn_rate,
        weight_decay=args.weight_decay)
    session.load_checkpoint(best_validation=True)
    network = session.network.derivative_network

    torch.no_grad()
    # arange input and state space
    step = 1e-3
    state = torch.arange(-1, 1, step).unsqueeze(1)
    input = torch.arange(-1, 1, step).unsqueeze(1).unsqueeze(2).repeat(1, state.shape[0], 1)
    time = dt * torch.arange(0, input.shape[0])

    # For each point calculate the derivative
    derivative_magnitude = torch.zeros(input.shape[1], state.shape[0])
    network.set_excitation_data(time, input)
    for i, t in enumerate(time):
        derivative_magnitude[:, i:i+1] = torch.abs(network(t, state))

    # Plot input vs state and display the magnitude of the learned derivative
    box = (-1, 1, -1, 1)
    plt.figure()
    plt.imshow(derivative_magnitude.detach().numpy(),
                extent=box, origin='lower')
    plt.colorbar()
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.savefig(session.run_directory / 'derivative.png', bbox_inches='tight', dpi=300)

    # TODO: Plot some points for reference
    # https://stackoverflow.com/questions/39727040/matplotlib-2d-plot-from-x-y-z-values

if __name__ == '__main__':
    main()
