"""Set up an NN architecture, run its training and test on the phaser data."""
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import CoreAudioML.networks as networks
import CoreAudioML.training as training
from NetworkTraining import NetworkTraining, create_dataset
from models import ODENet, ODENetDerivative, ResidualIntegrationNetworkRK4, BilinearBlock, ODENet2, ODENetDerivative2, ExcitationSecondsLinearInterpolation
from common import close_session, get_teacher_forcing_gate, attach_scheduler, get_device, load_checkpoint, save_args, test, get_run_name, get_method


def argument_parser():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        '--method',
        default='forward_euler',
        choices=[
            'LSTM',
            'STN',
            'ResIntRK4',
            'odeint_dopri5',
            'odeint_euler',
            'odeint_implicit_adams',
            'forward_euler',
            'trapezoid_rule'], help='(default: %(default)s)')
    ap.add_argument('--epochs', '-eps', type=int, default=20,
                    help='Max number of training epochs to run (default: %(default)s).')
    ap.add_argument('--batch_size', '-bs', type=int, default=256,
                    help='Training mini-batch size (default: %(default)s).')
    ap.add_argument('--learn_rate', '-lr', type=float, default=1e-3,
                    help='Initial learning rate (default: %(default)s).')
    ap.add_argument(
        '--cyclic_lr',
        '-y',
        type=float,
        default=None,
        help='If given, uses the cyclic learning rate schedule by Smith. Given learning rate parameter is used as the base learning rate, and max learning rate is this argument'
        's parameter.')
    ap.add_argument(
        '--one_cycle_lr',
        '-oc',
        type=float,
        default=None,
        help='If given, uses the one cycle learning rate schedule. Given learning rate parameter is used as the base learning rate, and max learning rate is this argument'
        's parameter.')
    ap.add_argument('--init_len', '-il', type=int, default=1000,
                    help='Number of sequence samples to process before starting weight updates (default: %(default)s).')
    ap.add_argument('--up_fr', '-uf', type=int, default=2048,
                    help='For recurrent models, number of samples to run in between updating network weights, i.e the '
                    'default argument updates every %(default)s samples (default: %(default)s).')
    ap.add_argument('--val_chunk', '-vs', type=int, default=0, help='Number of sequence samples to process'
                    'in each chunk of validation (default: %(default)s).')
    ap.add_argument('--test_chunk', '-tc', type=int, default=0, help='Number of sequence samples to process'
                    'in each chunk of test (default: %(default)s).')
    ap.add_argument('--checkpoint', '-c', type=str, default=None,
                    help='Load a checkpoint of the given architecture with the specified name.')
    ap.add_argument('--adjoint', '-adj', action='store_true',
                    help='Use the adjoint sensitivity method for backpropagation.')
    ap.add_argument('--name', '-n', type=str, default='', help='Set name for the run')
    ap.add_argument('--weight_decay', '-wd', type=float, default=0.0,
                    help='Weight decay argument for the Adam optimizer (default: %(default)s).')
    ap.add_argument(
        '--teacher_forcing',
        '-tf',
        nargs='?',
        const='always',
        default='never',
        choices=[
            'always',
            'never',
            'bernoulli'],
        help='Enable ground truth initialization of the first output sample in the minibatch. \n\'always\' uses teacher forcing in each minibatch;\n\'never\' never uses teacher forcing;\n\'bernoulli\' includes teacher forcing more rarely according to the fraction epochs passed.\n(default: %(default)s)')
    ap.add_argument('--test_sampling_rate', type=int, default=44100,
                    help='Sampling rate to use at test time. (default: %(default)s, same as in the training set).')
    ap.add_argument(
        '--dataset_name',
        default='FameSweetToneOffNoFb',
        choices=[
            'BehPhaserToneoff',
            'FameSweetToneOffNoFb'],
        help='Name of the dataset to use for modeling (default: %(default)s.')
    return ap


def get_architecture(args, dt):
    if args.method == 'LSTM':
        network = networks.SimpleRNN(unit_type="LSTM", hidden_size=16, skip=0, input_size=2)
    elif args.method == 'STN':
        network = StateTrajectoryNetworkFF()
    elif args.method == 'ResIntRK4':
        network = ResidualIntegrationNetworkRK4(nn.Sequential(BilinearBlock(input_size=3,
                                                              output_size=6,
                                                              latent_size=12),
                                                              BilinearBlock(input_size=6,
                                                              output_size=1,
                                                              latent_size=12)), dt=1.0)
    else:
        method = get_method(args)
        network = ODENet2(ODENetDerivative2(ExcitationSecondsLinearInterpolation()), method, dt)
    return network


def initialize_session(args):
    session = NetworkTraining()
    session.dataset = create_dataset(args.dataset_name, validation_frame_len=args.val_chunk, test_frame_len=args.test_chunk, test_sampling_rate=args.test_sampling_rate)
    session.epochs = args.epochs
    session.segments_in_a_batch = args.batch_size
    session.samples_between_updates = args.up_fr
    session.initialization_length = args.init_len
    session.enable_teacher_forcing = get_teacher_forcing_gate(args.teacher_forcing)
    session.loss = training.LossWrapper({'ESR': .5, 'DC': .5}, pre_filt=[1, -0.85])

    session.device = get_device()
    session.network = get_architecture(args, 1 / session.sampling_rate('train'))
    session.transfer_to_device()
    session.optimizer = torch.optim.Adam(
        session.network.parameters(),
        lr=args.learn_rate,
        weight_decay=args.weight_decay)
    attach_scheduler(args, session)

    model_directory = Path('phaser', 'runs', args.method.lower())

    load_checkpoint(args, session, model_directory)

    run_name = get_run_name(args.name)
    session.run_directory = model_directory / run_name

    save_args(session, args)

    return session



def main():
    args = argument_parser().parse_args()

    session = initialize_session(args)

    print('Training started.')
    try:
        session.run()
    except KeyboardInterrupt:
        print('Training interrupted, proceeding to test.')

    try:
        test(session)
    except KeyboardInterrupt:
        print('Test interrupted, quitting.')

    close_session(session)


if __name__ == '__main__':
    main()
