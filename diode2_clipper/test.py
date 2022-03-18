"""Set up an NN architecture, run its training and test on the second-order diode clipper data."""
import torch.nn as nn
from common import initialize_session, parse_args, test, close_session
from main import get_architecture


def main():
    args = parse_args()

    session = initialize_session('diode2_clipper', args, get_architecture)

    print('Test started.')
    try:
        test(session)
    except KeyboardInterrupt:
        print('Test interrupted, quitting.')

    close_session(session)


if __name__ == '__main__':
    main()
