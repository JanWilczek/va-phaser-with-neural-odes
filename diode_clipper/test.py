"""Set up an NN architecture, run its training and test on the diode clipper data."""
import torch.nn as nn
from common import initialize_session, argument_parser, test, close_session
from architectures import get_diode_clipper_architecture


def main():
    args = argument_parser().parse_args()

    session = initialize_session('diode_clipper', args, get_diode_clipper_architecture)

    try:
        test(session)
    except KeyboardInterrupt:
        print('Test interrupted, quitting.')

    close_session(session)


if __name__ == '__main__':
    main()
