"""Set up an NN architecture, run its training and test on the second-order diode clipper data."""
import torch.nn as nn
import CoreAudioML.networks as networks
from common import initialize_session, argument_parser, train_and_test, get_method
from architectures import ResidualIntegrationNetworkRK4, BilinearBlock, ODENet, ExcitationSecondsLinearInterpolation, get_nonlinearity, ScaledODENetFE, DerivativeMLPFE, DerivativeMLP2FE, DerivativeFEWithMemory, DerivativeMLPRK4, FlexibleStateTrajectoryNetwork, parse_layer_sizes, ScaledODENet, ScaledODENetRK4, ScaledODENetMidpoint


def get_architecture(args, dt):
    TARGET_SIZE = 2 # Second-order diode clipper has 2 states recorded in the dataset.
    output_size = max(TARGET_SIZE, args.state_size)
    
    if args.method == 'LSTM':
        network = networks.SimpleRNN(unit_type="LSTM", hidden_size=args.hidden_size, skip=0, input_size=1, output_size=TARGET_SIZE)
    elif args.method == 'ResIntRK4':
        network = ResidualIntegrationNetworkRK4(nn.Sequential(BilinearBlock(input_size=3,
                                                              output_size=6,
                                                              latent_size=12),
                                                              BilinearBlock(input_size=6,
                                                              output_size=TARGET_SIZE,
                                                              latent_size=12)), dt=1.0)
    elif args.method == 'STN':
        network = FlexibleStateTrajectoryNetwork(layer_sizes=parse_layer_sizes(args.layers_description),
                                                activation=get_nonlinearity(args),
                                                training_time_step=dt)
    elif args.method in ['ScaledODENetFE', 'ScaledODENetRK4', 'ScaledODENetMidpoint']:
        derivative_network_args = {'activation': get_nonlinearity(args),
                                    'excitation_size': 1,
                                    'output_size': output_size,
                                    'hidden_size': args.hidden_size}
        # Initialize the derivative network by name
        derivative_network = globals()[args.derivative_network](**derivative_network_args)
        network = globals()[args.method](derivative_network, int(1/dt), TARGET_SIZE)
    elif args.method.startswith('ScaledODENet'):
        method = get_method(args)
        derivative_network_args = {'excitation': ExcitationSecondsLinearInterpolation(),
                                    'activation': get_nonlinearity(args),
                                    'excitation_size': 1,
                                    'output_size': output_size,
                                    'hidden_size': args.hidden_size}
        # Initialize the derivative network by name
        derivative_network = globals()[args.derivative_network](**derivative_network_args)
        network = ScaledODENet(derivative_network, method, dt, TARGET_SIZE)
    else:
        method = get_method(args)
        derivative_network_args = {'excitation': ExcitationSecondsLinearInterpolation(),
                                    'activation': get_nonlinearity(args),
                                    'excitation_size': 1,
                                    'output_size': output_size,
                                    'hidden_size': args.hidden_size}
        # Initialize the derivative network by name
        derivative_network = globals()[args.derivative_network](**derivative_network_args)
        network = ODENet(derivative_network, method, dt, TARGET_SIZE)
    return network


def main():
    args = argument_parser().parse_args()

    session = initialize_session('diode2_clipper', args, get_architecture)

    if session.device != 'cuda':
        raise RuntimeError('CUDA not available.')

    train_and_test(session)


if __name__ == '__main__':
    main()
