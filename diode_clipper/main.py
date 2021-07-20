"""Set up an NN architecture, run its training and test on the diode clipper data."""
import CoreAudioML.networks as networks
from common import initialize_session, argument_parser, train_and_test, get_method
from models import StateTrajectoryNetworkFF
from architectures import ResidualIntegrationNetworkRK4, BilinearBlock, ODENet, DerivativeMLP, ExcitationSecondsLinearInterpolation


def get_architecture(args, dt):
    if args.method == 'LSTM':
        network = networks.SimpleRNN(unit_type="LSTM", hidden_size=8, skip=0)
    elif args.method == 'STN':
        network = StateTrajectoryNetworkFF(training_time_step=dt)
    elif args.method == 'ResIntRK4':
        network = ResidualIntegrationNetworkRK4(BilinearBlock(input_size=2, output_size=1, latent_size=6), dt)
    else:
        method = get_method(args)
        network = ODENet(DerivativeMLP(ExcitationSecondsLinearInterpolation(), nn.ReLU(), input_size=2, output_size=1, hidden_size=args.hidden_size), method, dt)
    return network


def main():
    args = argument_parser().parse_args()

    session = initialize_session('diode_clipper', args, get_architecture)

    train_and_test(session)


if __name__ == '__main__':
    main()
