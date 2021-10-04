from functools import partial
import torch.nn as nn
from torchdiffeq import odeint
from CoreAudioML.networks import SimpleRNN
from solvers import ForwardEuler
from architectures import ODENet, DerivativeMLP, DerivativeMLP2, StateTrajectoryNetwork, ExcitationSecondsLinearInterpolation, ResidualIntegrationNetworkRK4, BilinearBlock

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    dt = 1 / 44100
    networks = [SimpleRNN(unit_type="LSTM", hidden_size=8, skip=0),
                StateTrajectoryNetwork(training_time_step=dt),
                ResidualIntegrationNetworkRK4(BilinearBlock(input_size=2, output_size=1, latent_size=6), dt),
                ODENet(DerivativeMLP(ExcitationSecondsLinearInterpolation(), nn.ReLU(), excitation_size=1, output_size=1, hidden_size=100), ForwardEuler(), dt),
                ODENet(DerivativeMLP(ExcitationSecondsLinearInterpolation(), nn.ReLU(), excitation_size=1, output_size=1, hidden_size=9), ForwardEuler(), dt),
                ODENet(DerivativeMLP(ExcitationSecondsLinearInterpolation(), nn.ReLU(), excitation_size=1, output_size=1, hidden_size=9), partial(odeint, method='implicit_adams'), dt),
                SimpleRNN(unit_type="LSTM", hidden_size=16, skip=0, input_size=2),
                ODENet(DerivativeMLP2(ExcitationSecondsLinearInterpolation(), nn.SELU(), excitation_size=2, output_size=1, hidden_size=30), ForwardEuler(), dt),
                ODENet(DerivativeMLP2(ExcitationSecondsLinearInterpolation(), nn.SELU(), excitation_size=2, output_size=18, hidden_size=30), ForwardEuler(), dt),
                ODENet(DerivativeMLP2(ExcitationSecondsLinearInterpolation(), nn.SELU(), excitation_size=2, output_size=36, hidden_size=30), ForwardEuler(), dt)]

    print('Number of parameters of each model\n' \
          '====================================')

    for network in networks:
        print(f'{type(network).__name__}: {count_parameters(network)}')

if __name__ == '__main__':
    main()
