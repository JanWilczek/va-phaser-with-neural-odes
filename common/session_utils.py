import os
import json
import math
import torch
import traceback
import socket
from datetime import datetime
import argparse
from pathlib import Path
import CoreAudioML.dataset as dataset
from .resample import resample_test_files
from .NetworkTraining import NetworkTraining
from .loss import get_loss_function


def initialize_session(model_name, args, get_architecture):
    session = NetworkTraining()
    session.dataset = create_dataset(
        Path(model_name, 'data'),
        args.dataset_name,
        validation_frame_len=args.val_chunk,
        test_frame_len=args.test_chunk,
        test_sampling_rate=args.test_sampling_rate)
    session.epochs = args.epochs
    session.segments_in_a_batch = args.batch_size
    session.samples_between_updates = args.up_fr
    session.initialization_length = args.init_len
    session.validate_every = args.validate_every
    session.enable_teacher_forcing = get_teacher_forcing_gate(args.teacher_forcing)
    session.loss = get_loss_function(args.loss_function)

    session.device = get_device(args.device)
    session.network = get_architecture(args, 1 / session.sampling_rate('train'))
    session.transfer_to_device()
    session.optimizer = torch.optim.Adam(
        session.network.parameters(),
        lr=args.learn_rate,
        weight_decay=args.weight_decay)
    attach_scheduler(args, session)

    model_directory = Path(model_name, 'runs', args.dataset_name, args.method.lower())

    load_checkpoint(args, session, model_directory)

    run_name = get_run_name(args.name)
    session.run_directory = model_directory / run_name

    save_args(session, args)

    if args.save_sets:
        session.save_subsets()
    
    session.writer.add_text('Architecture', str(session.network))

    return session


def create_dataset(dataset_path: Path, dataset_name: str, train_frame_len=22050, validation_frame_len=0, test_frame_len=0, test_sampling_rate=44100):
    """Build a DataSet object.

    Parameters
    ----------
    dataset_path : Path
        Path to the folder containing the 'train', 'validation',
        and 'test' subfolders.
    dataset_name : str
        name of the dataset audio file.
        For example: in each of the 'train', 'validation',
        and 'test' subfolders you have files 'amplifier-input.wav' and 'amplifier-target.wav'. In this case supply the 'amplifier' string.
    train_frame_len : int, optional
        length of frames to split the train set into 
        (0 for one, long frame), by default 22050
    validation_frame_len : int, optional
        length of frames to split the validation set into
        (0 for one, long frame), by default 0
    test_frame_len : int, optional
        length of frames to split the test set into
        (0 for one, long frame), by default 0
    test_sampling_rate : int, optional
        sampling rate to use at test time.
        If different than the default (44100) the test files
        will be resampled and saved under a new name, i.e.,
        {dataset_path}/test/{dataset_name}{test_sampling_rate}Hz-input.wav
        {dataset_path}/test/{dataset_name}{test_sampling_rate}Hz-target.wav

    Returns
    -------
    DataSet
        the created DataSet object
    """
    d = dataset.DataSet(data_dir=str(dataset_path))
    d.name = dataset_name

    d.create_subset('train', frame_len=train_frame_len)
    d.create_subset('validation', frame_len=validation_frame_len)
    d.create_subset('test', frame_len=test_frame_len)

    d.load_file(os.path.join('train', dataset_name) , 'train')
    d.load_file(os.path.join('validation', dataset_name), 'validation')
    
    test_filename = os.path.join('test', dataset_name)
    if test_sampling_rate != d.subsets['train'].fs:
        test_filename = resample_test_files(dataset_path, test_filename, test_sampling_rate)
    d.load_file(test_filename, set_names='test')

    return d


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
    elif args.exponential_lr is not None:
        session.scheduler = torch.optim.lr_scheduler.ExponentialLR(session.optimizer,
                                                                   gamma=math.exp(math.log(args.exponential_lr / args.learn_rate) / (session.epochs * session.minibatch_count)),
                                                                   last_epoch = max(-1, session.minibatch_count * (session.epoch - 1)))
    elif args.step_lr is not None:
        session.scheduler = torch.optim.lr_scheduler.StepLR(session.optimizer,
                                                            step_size=args.step_lr[0],
                                                            gamma=args.step_lr[1],
                                                            last_epoch=(session.epoch - 1))


def load_checkpoint(args, session, model_directory):
    if args.checkpoint is not None:
        session.run_directory = model_directory / args.checkpoint
        session.load_checkpoint(best_validation=args.best_validation)
        if session.scheduler is None or args.overwrite_lr:
            learn_rate = args.overwrite_lr if args.overwrite_lr else args.learn_rate
            # Scheduler's learning rate is not loaded but overwritten
            for param_group in session.optimizer.param_groups:
                param_group['lr'] = learn_rate


def get_device(device):
    return 'cuda' if torch.cuda.is_available() and device == 'gpu' else 'cpu'


def save_args(session, args):
    save_json(vars(args), session.run_directory / 'args.json')
    session.writer.add_text('Command line arguments', json.dumps(vars(args)))


def test(session):
    session.device = 'cpu'
    # Load the model performing best on the validation set for test
    try:
        session.load_checkpoint(best_validation=True)
    except FileNotFoundError:
        print("No best validation model found. Staying with the current setup.")
    test_output, test_loss = session.test()
    print(f'Test loss: {test_loss}')
    session.writer.add_scalar('Loss/test', test_loss, session.epochs)
    
    # Flatten the first channel of the output properly to obtain one long frame
    test_audio = test_output.permute(1, 0, 2)[:, :, 0].flatten()[None, :]
    test_output_path = (session.run_directory / 'test_output.wav').resolve()
    session.save_audio(test_output_path, test_audio, session.sampling_rate('test'))


def get_teacher_forcing_gate(teacher_forcing_description):
    description_to_gate = {'always': lambda epoch_progress: True,
                           'never': lambda epoch_progress: False,
                           'bernoulli': lambda epoch_progress: torch.bernoulli(torch.Tensor([1 - epoch_progress]))
                           }
    return description_to_gate[teacher_forcing_description]


def log_memory_usage(session):
    if torch.cuda.is_available():
        BYTES_IN_MEGABYTE = 2 ** 20
        session.writer.add_scalar('Maximum GPU memory usage [MB]',
                                  torch.cuda.max_memory_allocated('cuda') / BYTES_IN_MEGABYTE,
                                  session.epochs)

def close_session(session):
    log_memory_usage(session)
    session.writer.close()

def save_json(json_data, filepath):
    with open(filepath, 'w') as f:
        json.dump(json_data, f, indent=4)

def get_run_name(suffix=''):
    name = datetime.now().strftime(r"%B%d_%H-%M-%S") + f'_{socket.gethostname()}'
    if len(suffix) > 0:
        name += '_' + suffix
    return name


def parse_args():
    config_file_parser = argparse.ArgumentParser(add_help=False)
    config_file_parser.add_argument('-cf', '--config_file', default=None)
    args, unknown = config_file_parser.parse_known_args()
    parser = argument_parser(parents=[config_file_parser], add_help=False)
    if args.config_file:
        config = json.load(open(args.config_file))
        parser.set_defaults(**config)
    return parser.parse_args()


def argument_parser(*args, **kwargs):
    ap = argparse.ArgumentParser(*args, **kwargs, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        '--method',
        choices=[
            'LSTM',
            'STN',
            'ResIntRK4',
            'odeint_dopri5',
            'odeint_rk4',
            'odeint_euler',
            'odeint_implicit_adams',
            'forward_euler',
            'trapezoid_rule',
            'ScaledODENetFE',
            'ScaledODENetRK4',
            'ScaledODENetMidpoint',
            'ScaledODENet_dopri5',
            'ScaledODENet_rk4',
            'ScaledODENet_euler',
            'ScaledODENet_implicit_adams'], 
            required=False,
            help='Method to use for numerical integration of the differential equation.')
    ap.add_argument('--epochs', '-eps', type=int, required=False,
                    help='Max number of training epochs to run.')
    ap.add_argument('--batch_size', '-bs', type=int, required=False,
                    help='Training mini-batch size.')
    ap.add_argument('--learn_rate', '-lr', type=float, default=1e-3,
                    help='Initial learning rate (default: %(default)s).')
    ap.add_argument(
        '--cyclic_lr',
        '-y',
        type=float,
        default=None,
        help='If given, uses the cyclic learning rate schedule by Smith. Given learning rate '
             'parameter is used as the base learning rate, and max learning rate is this argument'
        's parameter.')
    ap.add_argument(
        '--one_cycle_lr',
        '-oc',
        type=float,
        default=None,
        help='If given, uses the one cycle learning rate schedule. Given learning rate parameter '
             'is used as the base learning rate, and max learning rate is this argument'
        's parameter.')
    ap.add_argument(
        '--exponential_lr',
        '-elr',
        type=float,
        default=None,
        help='If given, uses the exponential learning rate schedule: exponentially decreases '
             'the learning rate from the one given in the learn_rate argument to the one '
             'specified in this argument.'
    )
    ap.add_argument(
        '--step_lr',
        '-slr',
        nargs=2,
        type=float,
        default=None,
        help='If given, uses the step learning rate schedule: every step_size epochs '
             '(first parameter) multiplies the learning rate by gamma (second parameter).'
    )
    ap.add_argument('--overwrite_lr',
                    '-olr',
                    type=float,
                    default=None,
                    help='If given, will overwrite the instantaneous learning rate. Can be used to '
                         'rapidly decrease the learning rate if the training diverged.')
    ap.add_argument('--init_len', '-il', type=int, default=0,
                    help='Number of sequence samples to process before starting weight updates '
                         '(default: %(default)s).')
    ap.add_argument('--up_fr', '-uf', type=int, default=2048,
                    help='For recurrent models, number of samples to run in between updating '
                         'network weights, i.e the '
                    'default argument updates every %(default)s samples (default: %(default)s).')
    ap.add_argument('--val_chunk', '-vs', type=int, default=22050, help='Number of sequence samples to process'
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
        help='Enable ground truth initialization of the first output sample in the minibatch. '
             '\n\'always\' uses teacher forcing in each minibatch;\n\'never\' never uses '
             'teacher forcing;\n\'bernoulli\' includes teacher forcing more rarely according '
             'to the fraction epochs passed.\n(default: %(default)s)')
    ap.add_argument(
        '--hidden_size',
        default=100,
        type=int,
        help='The size of the hidden layers (model-dependent) (default: %(default)s).')
    ap.add_argument('--test_sampling_rate', type=int, default=44100,
                    help='Sampling rate to use at test time. (default: %(default)s, '
                         'same as in the training set).')
    ap.add_argument(
        '--save_sets',
        action='store_true',
        help='If set, the training, validation and test sets will be saved in the output folder.')
    ap.add_argument(
        '--dataset_name',
        help='Name of the dataset to use for modeling.',
        required=False)
    ap.add_argument('--nonlinearity', default='ReLU', help='Name of the torch.nn nonlinearity to use in the ODENet derivative network if that method is used (default: %(default)s).')
    ap.add_argument('--validate_every', default=1, type=int, help='Number of epochs to calculate validation loss after (default: %(default)s).')
    ap.add_argument('--state_size', default=1, type=int, help='Number of elements of the state vector of the dynamical system. The first element is always taken as the audio output of the system (default: %(default)s).')
    ap.add_argument('--loss_function', default='ESR_DC_prefilter', help='Loss function to use during training (default: %(default)s). Possible choices:' \
        '\n{:<20}\t{:<20}'.format('ESR_DC_prefilter', '0.5 * ESR(prefiltered_output, prefiltered_target) + 0.5 * DCloss(prefiltered_output, prefiltered_target)') + \
        '\n{:<20}\t{:<20}'.format('L1_STFT', 'L1(STFT_output, STFT_target).') + \
        '\n{:<20}\t{:<20}'.format('L2_STFT', 'L2(STFT_output, STFT_target).') + \
        '\nlog_spectral_distance')
    ap.add_argument('--derivative_network', default='DerivativeMLP2', help='Derivative network to use in case of ODENet.')
    ap.add_argument('--layers_description', default=None, type=str, help='Description of layers of the network. For example, "1x2x3" denotes an MLP with layers of size 1, 2, and 3, with all but the last one having the activation function applied at their output.')
    ap.add_argument('--best_validation', action='store_true', help='If provided the best validation checkpoint is loaded. Otherwise, the last checkpoint is loaded.')
    ap.add_argument('--device', '-d', choices=['cpu', 'gpu'], default='gpu',
                    help='Device to run the training on (default: %(default)).')
    return ap


def train_and_test(session):
    print('Training started.')
    try:
        session.run()
    except KeyboardInterrupt:
        print('Training interrupted.')
    except RuntimeError as err:
        error_message = f'RuntimeError: {err}\n{session.debug_string()}'
        print(error_message)
        traceback.print_last()

    print('Test started.')
    try:
        test(session)
    except KeyboardInterrupt:
        print('Test interrupted, quitting.')

    close_session(session)
