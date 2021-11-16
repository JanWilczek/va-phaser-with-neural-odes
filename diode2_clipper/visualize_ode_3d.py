import subprocess
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    
def plot4d(data):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    mask = data > 0.01
    idx = np.arange(int(np.prod(data.shape)))
    x, y, z = np.unravel_index(idx, data.shape)
    ax.scatter(x, y, z, c=data.flatten(), s=10.0 * mask, edgecolor="face", alpha=0.2, marker="o", cmap="magma", linewidth=0)
    plt.tight_layout()
    plt.savefig("test_scatter_4d.png", dpi=250)
    plt.close(fig)

def main():
    # setup_pyplot_for_latex()
    
    # Load model
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
    dt = 1 / 44100
    session.network = get_architecture(args, dt)
    session.optimizer = torch.optim.Adam(
        session.network.parameters(),
        lr=args.learn_rate,
        weight_decay=args.weight_decay)
    session.load_checkpoint(best_validation=True)
    network = session.network.derivative_network

    torch.no_grad()
    # Arrange input and state space
    step = 1e-2
    state1 = torch.arange(-1, 1, step)
    state2 = torch.arange(-2, 2, step)
    input = torch.arange(-1, 1, step)
    
    figures_dir = session.run_directory / 'figures'
    
    derivative_magnitude = torch.zeros(input.shape[0], state1.shape[0], state2.shape[0], 2)
    
    network.excitation = input.unsqueeze(1).unsqueeze(2)
    
    time = torch.arange(0, input.shape[0])
    
    x = torch.zeros((state1.shape[0] * state2.shape[0] * input.shape[0],))
    y = torch.zeros_like(x)
    z = torch.zeros_like(x)
    color_value = torch.zeros((x.shape[0], 2))
    n = 0
    
    for j, state1_value in enumerate(state1):
        for k, state2_value in enumerate(state2):
            # For each point calculate the derivative
            for i, t in enumerate(time):
                derivative_magnitude[i, j, k] = torch.abs(network(t, torch.tensor([state1_value, state2_value]).unsqueeze(0)))
                
                x[n] = input[i]
                y[n] = state1_value
                z[n] = state2_value
                color_value[n] = derivative_magnitude[i, j, k]
                n += 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pnt3d = ax.scatter(x.detach().numpy(), y.detach().numpy(), z.detach().numpy(), c=color_value[:, 1].detach().numpy())
    cbar = plt.colorbar(pnt3d)
    cbar.set_label("State 1 or 2 magnitude")
    plt.savefig(figures_dir / 'ode_3d.png', bbox_inches='tight', dpi=300)
    
    plot4d(derivative_magnitude[..., 0].detach().numpy())
    
    # plot_ode(derivative_magnitude[..., 0].detach().numpy(), 'dy1')
    # plt.savefig(figures_dir / f'dy1_{state2:.2f}.png', bbox_inches='tight', dpi=300)
    # plot_ode(derivative_magnitude[..., 1].detach().numpy(), 'dy2')
    # plt.savefig(figures_dir / f'dy2_{state2:.2f}.png', bbox_inches='tight', dpi=300)
    # save_tikz(session.run_directory / 'ode_derivative')

    # WARNING: This function is dangerous, may delete your valuable content.
    # clean_up(session.run_directory)

if __name__ == '__main__':
    main()
