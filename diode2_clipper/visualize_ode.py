import subprocess
import argparse
import json
import matplotlib.pyplot as plt
import torch
from common import NetworkTraining, argument_parser, setup_pyplot_for_latex, save_tikz
from .main import get_architecture


def clean_up(run_path):
    bash_command = f"find {run_path} -ctime -1 -type f -not -name *.png -print -delete".split()   # This assumes that the script is running for max 60 seconds
    process = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output)

def plot_ode(derivative_magnitude, ylabel='Output'):
    box = (-1, 1, -1, 1)
    plt.figure()
    plt.imshow(derivative_magnitude, extent=box, origin='lower')
    plt.colorbar()
    plt.xlabel('Input')
    plt.ylabel(ylabel)

def load_session():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_path', help='Path to the directory with the run to analyze model from.')
    session = NetworkTraining()
    session.run_directory = ap.parse_args().run_path
    with open(session.run_directory / 'args.json', 'r') as f:
        args_dict = json.load(f)
    args = argument_parser().parse_args(['--method', args_dict['method'],
                                        '-eps', str(args_dict['epochs']),
                                        '-bs', str(args_dict['batch_size']),
                                        '--dataset_name', 'diodeclip'])
    args.__dict__.update(args_dict)
    dt = 1 / 44100 # This is dataset's time step size
    session.network = get_architecture(args, dt)
    session.optimizer = torch.optim.Adam(
        session.network.parameters(),
        lr=args.learn_rate,
        weight_decay=args.weight_decay)
    session.load_checkpoint(best_validation=True)
    return session
    
def main():
    setup_pyplot_for_latex()
    
    session = load_session()
    network = session.network.derivative_network
    figures_dir = session.run_directory / 'figures'
    figures_dir.mkdir(exist_ok=True)

    torch.no_grad()
    
    # Arrange input and state space
    step = 1e-3 # Step with which to sample the space (input and individual states' values)
    state1 = torch.arange(-1, 1, step).unsqueeze(1)
    input = torch.arange(-1, 1, step).unsqueeze(1).unsqueeze(2).repeat(1, state1.shape[0], 1)
    
    for state2 in [-1.5, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
        state_full = torch.cat((state1, state2 * torch.ones_like(state1)), dim=1)
        time = torch.arange(0, input.shape[0])

        # For each point calculate the derivative
        derivative_magnitude = torch.zeros(input.shape[1], state1.shape[0], 2)
        network.excitation = input
        for i, t in enumerate(time):
            derivative_magnitude[:, i, :] = torch.abs(network(t, state_full))

        # Plot input vs state and display the magnitude of the learned derivative
        test_img = torch.zeros((10, 10))
        test_img[0, :] = -1 # bottom row
        test_img[:, 0] = 1 # leftmost column

        plot_ode(derivative_magnitude[..., 0].detach().numpy(), 'y1')
        plt.savefig(figures_dir / f'dy1_{state2:.2f}.png', bbox_inches='tight', dpi=300)
        save_tikz(figures_dir / f'dy1_{state2:.2f}.tex')
        plot_ode(derivative_magnitude[..., 1].detach().numpy(), 'y1')
        plt.savefig(figures_dir / f'dy2_{state2:.2f}.png', bbox_inches='tight', dpi=300)
        save_tikz(figures_dir / f'dy2_{state2:.2f}.tex')

        # WARNING: This function is dangerous, may delete your valuable content.
        # clean_up(session.run_directory)

if __name__ == '__main__':
    main()
