"""Set up an NN architecture, run its training and test on the phaser data."""
import torch.nn as nn
import CoreAudioML.networks as networks
from common import initialize_session, argument_parser, train_and_test, get_method
from architectures import ResidualIntegrationNetworkRK4, BilinearBlock, ODENet, DerivativeMLP, DerivativeMLP2, ExcitationSecondsLinearInterpolation, get_nonlinearity


def get_architecture(args, dt):
    if args.method == 'LSTM':
        network = networks.SimpleRNN(unit_type="LSTM", hidden_size=16, skip=0, input_size=2, output_size=args.state_size)
    elif args.method == 'ResIntRK4':
        network = ResidualIntegrationNetworkRK4(nn.Sequential(BilinearBlock(input_size=3,
                                                              output_size=6,
                                                              latent_size=12),
                                                              BilinearBlock(input_size=6,
                                                              output_size=1,
                                                              latent_size=12)), dt=1.0)
    else:
        method = get_method(args)
        derivative_network_args = {'excitation': ExcitationSecondsLinearInterpolation(),
                                    'activation': get_nonlinearity(args),
                                    'excitation_size': 2,
                                    'output_size': args.state_size,
                                    'hidden_size': args.hidden_size}
        # Initialize the derivative network by name
        derivative_network = globals()[args.derivative_network](**derivative_network_args)
        network = ODENet(derivative_network, method, dt)
    return network


def main():
    args = argument_parser().parse_args()

    session = initialize_session('phaser', args, get_architecture)

    train_and_test(session)


if __name__ == '__main__':
    main()
