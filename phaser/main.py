"""Set up an NN architecture, run its training and test on the phaser data."""
import argparse
import json
from functools import partial
from pathlib import Path
import torch
import torch.nn as nn
import torchaudio
from torchdiffeq import odeint, odeint_adjoint
import CoreAudioML.networks as networks
import CoreAudioML.training as training
from NetworkTraining import NetworkTraining, get_run_name, create_dataset, save_json
from models import ODENet, ODENetDerivative, ResidualIntegrationNetworkRK4, BilinearBlock, ODENet2, ODENetDerivative2, ExcitationSecondsLinearInterpolation
from solvers import ForwardEuler, trapezoid_rule


def argument_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('method', nargs='+')
    ap.add_argument('--epochs', '-eps', type=int, default=20, help='Max number of training epochs to run')
    ap.add_argument('--batch_size', '-bs', type=int, default=256, help='Training mini-batch size')
    ap.add_argument('--learn_rate', '-lr', type=float, default=1e-3, help='Initial learning rate')
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
                    help='Number of sequence samples to process before starting weight updates')
    ap.add_argument('--up_fr', '-uf', type=int, default=2048,
                    help='For recurrent models, number of samples to run in between updating network weights, i.e the '
                    'default argument updates every 1000 samples')
    ap.add_argument('--val_chunk', '-vs', type=int, default=0, help='Number of sequence samples to process'
                    'in each chunk of validation ')
    ap.add_argument('--test_chunk', '-tc', type=int, default=0, help='Number of sequence samples to process'
                    'in each chunk of test')
    ap.add_argument('--checkpoint', '-c', type=str, default=None,
                    help='Load a checkpoint of the given architecture with the specified name.')
    ap.add_argument('--adjoint', '-adj', action='store_true',
                    help='Use the adjoint sensitivity method for backpropagation.')
    ap.add_argument('--name', '-n', type=str, default='', help='Set name for the run')
    ap.add_argument('--weight_decay', '-wd', type=float, default=0.0,
                    help='Weight decay argument for the Adam optimizer.')
    return ap


def get_method(args):
    CUSTOM_SOLVERS = {'forward_euler': ForwardEuler(),
                      'trapezoid_rule': trapezoid_rule}
    if args.method[0] == 'odenet':
        if args.adjoint:
            return partial(odeint_adjoint, method=args.method[1])
        else:
            return partial(odeint, method=args.method[1])
    else:
        return CUSTOM_SOLVERS[args.method[0]]


def get_architecture(args, dt):
    if args.method[0] == 'LSTM':
        network = networks.SimpleRNN(unit_type="LSTM", hidden_size=16, skip=0, input_size=2)
    elif args.method[0] == 'STN':
        network = StateTrajectoryNetworkFF()
    elif args.method[0] == 'ResIntRK4':
        network = ResidualIntegrationNetworkRK4(nn.Sequential(BilinearBlock(input_size=3,
                                                              output_size=6,
                                                              latent_size=12),
                                                              BilinearBlock(input_size=6,
                                                              output_size=1,
                                                              latent_size=12)), dt)
    else:
        method = get_method(args)
        network = ODENet2(ODENetDerivative2(ExcitationSecondsLinearInterpolation(dt)), method, dt)
    return network


def attach_scheduler(args, session):
    if args.one_cycle_lr is not None:
        session.scheduler = torch.optim.lr_scheduler.OneCycleLR(session.optimizer,
                                                                max_lr=args.one_cycle_lr,
                                                                div_factor=(args.one_cycle_lr / args.learn_rate),
                                                                final_div_factor=20,
                                                                epochs=(session.epochs - session.epoch),
                                                                steps_per_epoch=session.minibatch_count,
                                                                last_epoch=(session.epoch - 1),
                                                                cycle_momentum=False)
    elif args.cyclic_lr is not None:
        session.scheduler = torch.optim.lr_scheduler.CyclicLR(session.optimizer,
                                                              base_lr=args.learn_rate,
                                                              max_lr=args.cyclic_lr,
                                                              step_size_up=250,
                                                              last_epoch=(session.epoch - 1),
                                                              cycle_momentum=False)


def load_checkpoint(args, session, model_directory):
    # Untested
    if args.checkpoint is not None:
        session.run_directory = model_directory / args.checkpoint
        try:
            session.load_checkpoint(best_validation=True)
        except BaseException:
            print("Failed to load a best validation checkpoint. Reverting to the last checkpoint.")
            session.load_checkpoint(best_validation=False)
        if session.scheduler is None:
            # Scheduler's learning rate is not loaded but overwritten
            for param_group in session.optimizer.param_groups:
                param_group['lr'] = args.learn_rate


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def save_args(session, args):
    save_json(vars(args), session.run_directory / 'args.json')
    session.writer.add_text('Command line arguments', json.dumps(vars(args)))


def test(session):
    session.device = 'cpu'
    test_output, test_loss = session.test()
    print(f'Test loss: {test_loss}')
    session.writer.add_scalar('Loss/test', test_loss, session.epochs)

    test_output_path = (session.run_directory / 'test_output.wav').resolve()
    torchaudio.save(test_output_path, test_output[None, :], session.sampling_rate)


def log_memory_usage(session):
    if torch.cuda.is_available():
        session.writer.add_scalar('Maximum GPU memory usage [MB]',
                                  torch.cuda.max_memory_allocated('cuda') / (2 ** 20),
                                  session.epochs)


def main():
    args = argument_parser().parse_args()

    session = NetworkTraining()
    session.dataset = create_dataset(validation_frame_len=args.val_chunk, test_frame_len=args.test_chunk)
    session.epochs = args.epochs
    session.segments_in_a_batch = args.batch_size
    session.samples_between_updates = args.up_fr
    session.initialization_length = args.init_len
    session.loss = training.LossWrapper({'ESR': .5, 'DC': .5}, pre_filt=[-0.85, 1])

    session.device = get_device()
    session.network = get_architecture(args, 1/session.sampling_rate)
    session.transfer_to_device()
    session.optimizer = torch.optim.Adam(
        session.network.parameters(),
        lr=args.learn_rate,
        weight_decay=args.weight_decay)
    attach_scheduler(args, session)

    model_directory = Path('phaser', 'runs', args.method[0].lower())

    load_checkpoint(args, session, model_directory)

    run_name = get_run_name(args.name)
    session.run_directory = model_directory / run_name

    save_args(session, args)

    session.run()

    test(session)

    log_memory_usage(session)

    session.writer.close()


if __name__ == '__main__':
    main()
