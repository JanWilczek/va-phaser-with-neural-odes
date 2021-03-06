from functools import partial
import torch.nn as nn
from torchdiffeq import odeint
from CoreAudioML.networks import SimpleRNN
from solvers import ForwardEuler
from architectures import ODENet, DerivativeMLP, DerivativeMLP2, StateTrajectoryNetwork, ExcitationSecondsLinearInterpolation, ResidualIntegrationNetworkRK4, BilinearBlock, ScaledODENetFE, DerivativeMLPFE, DerivativeMLP2FE, FlexibleStateTrajectoryNetwork, parse_layer_sizes

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    dt = 1 / 44100
    networks = [SimpleRNN(unit_type="LSTM", hidden_size=8, skip=0),
                StateTrajectoryNetwork(training_time_step=dt),
                ResidualIntegrationNetworkRK4(BilinearBlock(input_size=2, output_size=1, latent_size=6), dt),
                ODENet(DerivativeMLP(ExcitationSecondsLinearInterpolation(), nn.ReLU(), excitation_size=1, output_size=1, hidden_size=100), ForwardEuler(), dt, 1),
                ODENet(DerivativeMLP(ExcitationSecondsLinearInterpolation(), nn.ReLU(), excitation_size=1, output_size=1, hidden_size=9), ForwardEuler(), dt, 1),
                ODENet(DerivativeMLP(ExcitationSecondsLinearInterpolation(), nn.ReLU(), excitation_size=1, output_size=1, hidden_size=9), partial(odeint, method='implicit_adams'), dt, 1),
                SimpleRNN(unit_type="LSTM", hidden_size=16, skip=0, input_size=2),
                ODENet(DerivativeMLP2(ExcitationSecondsLinearInterpolation(), nn.SELU(), excitation_size=2, output_size=1, hidden_size=30), ForwardEuler(), dt, 1),
                ODENet(DerivativeMLP2(ExcitationSecondsLinearInterpolation(), nn.SELU(), excitation_size=2, output_size=18, hidden_size=30), ForwardEuler(), dt, 1),
                ODENet(DerivativeMLP2(ExcitationSecondsLinearInterpolation(), nn.SELU(), excitation_size=2, output_size=36, hidden_size=30), ForwardEuler(), dt, 1),
                # Second-order diode clipper models
                SimpleRNN(unit_type="LSTM", hidden_size=16, input_size=1, output_size=2, skip=0),
                SimpleRNN(unit_type="LSTM", hidden_size=32, input_size=1, output_size=2, skip=0),
                ScaledODENetFE(DerivativeMLPFE(activation=nn.Softsign(), excitation_size=1, output_size=2, hidden_size=10), int(1/dt), 2),
                ScaledODENetFE(DerivativeMLPFE(activation=nn.Softsign(), excitation_size=1, output_size=2, hidden_size=20), int(1/dt), 2),
                ScaledODENetFE(DerivativeMLPFE(activation=nn.Softsign(), excitation_size=1, output_size=2, hidden_size=30), int(1/dt), 2),
                ScaledODENetFE(DerivativeMLPFE(activation=nn.Softsign(), excitation_size=1, output_size=2, hidden_size=40), int(1/dt), 2),
                FlexibleStateTrajectoryNetwork(layer_sizes=parse_layer_sizes('3x20x20x20x2'),
                                                activation=nn.Tanh(),
                                                training_time_step=dt),
                FlexibleStateTrajectoryNetwork(layer_sizes=parse_layer_sizes('3x30x30x2'),
                                                activation=nn.Tanh(),
                                                training_time_step=dt)]

    print('Number of parameters of each model\n' \
          '====================================')

    for network in networks:
        print(f'{type(network).__name__}: {count_parameters(network)}')

if __name__ == '__main__':
    main()
