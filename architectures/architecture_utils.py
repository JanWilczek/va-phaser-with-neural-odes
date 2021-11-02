import torch.nn as nn
import CoreAudioML.networks as networks
from common import get_method
from . import ResidualIntegrationNetworkRK4, BilinearBlock, ODENet, DerivativeMLP, ExcitationSecondsLinearInterpolation, StateTrajectoryNetwork, ScaledODENetFE, DerivativeMLPFE, DerivativeMLP2FE, DerivativeFEWithMemory


def get_nonlinearity(args):
    return getattr(nn, args.nonlinearity)()


def get_diode_clipper_architecture(args, dt):
    if args.method == 'LSTM':
        network = networks.SimpleRNN(unit_type="LSTM", hidden_size=args.hidden_size, skip=0)
    elif args.method == 'STN':
        network = StateTrajectoryNetwork(training_time_step=dt)
    elif args.method == 'ResIntRK4':
        network = ResidualIntegrationNetworkRK4(BilinearBlock(input_size=2, output_size=1, latent_size=args.hidden_size), dt)
    elif args.method == 'ScaledODENetFE':
        derivative_network_args = {'activation': get_nonlinearity(args),
                                    'excitation_size': 1,
                                    'output_size': args.state_size,
                                    'hidden_size': args.hidden_size}
        # Initialize the derivative network by name
        derivative_network = globals()[args.derivative_network](**derivative_network_args)
        network = ScaledODENetFE(derivative_network, int(1/dt))
    else:
        method = get_method(args)
        network = ODENet(DerivativeMLP(ExcitationSecondsLinearInterpolation(), get_nonlinearity(args), excitation_size=1, output_size=args.state_size, hidden_size=args.hidden_size), method, dt)
    return network

def parse_layer_sizes(description):
    """
    Returns a list of integers corresponding to the description.

    Example:
    description is "1x2x3" string => [1, 2, 3] is returned.
    """
    return [int(size) for size in description.split('x')]
