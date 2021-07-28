import torch.nn as nn
import CoreAudioML.networks as networks
from . import ResidualIntegrationNetworkRK4, BilinearBlock, ODENet, DerivativeMLP, ExcitationSecondsLinearInterpolation, StateTrajectoryNetwork


def get_nonlinearity(args):
    return getattr(nn, args.nonlinearity)()


def get_diode_clipper_architecture(args, dt):
    if args.method == 'LSTM':
        network = networks.SimpleRNN(unit_type="LSTM", hidden_size=args.hidden_size, skip=0)
    elif args.method == 'STN':
        network = StateTrajectoryNetwork(training_time_step=dt)
    elif args.method == 'ResIntRK4':
        network = ResidualIntegrationNetworkRK4(BilinearBlock(input_size=2, output_size=1, latent_size=args.hidden_size), dt)
    else:
        method = get_method(args)
        network = ODENet(DerivativeMLP(ExcitationSecondsLinearInterpolation(), get_nonlinearity(args), excitation_size=1, output_size=args.state_size, hidden_size=args.hidden_size), method, dt)
    return network
