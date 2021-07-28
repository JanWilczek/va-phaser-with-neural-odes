"""Set up an NN architecture, run its training and test on the diode clipper data."""
import torch.nn as nn
from common import initialize_session, argument_parser, train_and_test, get_method
from architectures import get_diode_clipper_architecture


def main():
    args = argument_parser().parse_args()

    session = initialize_session('diode_clipper', args, get_diode_clipper_architecture)

    train_and_test(session)


if __name__ == '__main__':
    main()
