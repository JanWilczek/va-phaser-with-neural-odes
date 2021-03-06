"""Set up an NN architecture, run its training and test on the phaser data."""
import torch.nn as nn
import CoreAudioML.networks as networks
from common import initialize_session, argument_parser, train_and_test, get_method
from architectures import ResidualIntegrationNetworkRK4, BilinearBlock, ODENet, DerivativeMLP, DerivativeMLP2, SingleLinearLayer, ScaledSingleLinearLayer, DerivativeLSTM, DerivativeWithMemory, ExcitationSecondsLinearInterpolation, get_nonlinearity, ScaledODENetFE, DerivativeMLPFE, DerivativeMLP2FE, DerivativeFEWithMemory


def get_architecture(args, dt):
    if 'AllpassStates' in args.dataset_name:
        target_size = 11 # AllpassStates dataset contains 11 states at each time point
    else:
        target_size = 1 # Phaser has 1 state recorded in the dataset.
    output_size = max(target_size, args.state_size)
    
    if args.method == 'LSTM':
        network = networks.SimpleRNN(unit_type="LSTM", hidden_size=16, skip=0, input_size=2, output_size=target_size)
    elif args.method == 'ResIntRK4':
        network = ResidualIntegrationNetworkRK4(nn.Sequential(BilinearBlock(input_size=3,
                                                              output_size=6,
                                                              latent_size=12),
                                                              BilinearBlock(input_size=6,
                                                              output_size=target_size,
                                                              latent_size=12)), dt=1.0)
    elif args.method == 'ScaledODENetFE':
        derivative_network_args = {'activation': get_nonlinearity(args),
                                    'excitation_size': 2,
                                    'output_size': output_size,
                                    'hidden_size': args.hidden_size}
        # Initialize the derivative network by name
        derivative_network = globals()[args.derivative_network](**derivative_network_args)
        network = ScaledODENetFE(derivative_network, int(1/dt), target_size)
    else:
        method = get_method(args)
        derivative_network_args = {'excitation': ExcitationSecondsLinearInterpolation(),
                                    'activation': get_nonlinearity(args),
                                    'excitation_size': 2,
                                    'output_size': output_size,
                                    'hidden_size': args.hidden_size}
        # Initialize the derivative network by name
        derivative_network = globals()[args.derivative_network](**derivative_network_args)
        network = ODENet(derivative_network, method, dt, target_size)
    return network


def main():
    args = argument_parser().parse_args()

    session = initialize_session('phaser', args, get_architecture)

    train_and_test(session)


if __name__ == '__main__':
    main()
