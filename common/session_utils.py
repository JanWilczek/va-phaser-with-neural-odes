import os
import json
import torch
import torchaudio
import socket
from datetime import datetime
import argparse
from pathlib import Path
import CoreAudioML.dataset as dataset
from CoreAudioML import training
from .resample import resample_test_files
from .NetworkTraining import NetworkTraining


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
    session.loss = training.LossWrapper({'ESR': .5, 'DC': .5}, pre_filt=[1, -0.85])

    session.device = get_device()
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


def load_checkpoint(args, session, model_directory):
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
    torchaudio.save(test_output_path, test_output[None, :], session.sampling_rate('test'))


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

def argument_parser():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
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
            'trapezoid_rule'], 
            required=True,
            help='Method to use for numerical integration of the differential equation.')
    ap.add_argument('--epochs', '-eps', type=int, required=True,
                    help='Max number of training epochs to run.')
    ap.add_argument('--batch_size', '-bs', type=int, required=True,
                    help='Training mini-batch size.')
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
    ap.add_argument('--init_len', '-il', type=int, default=0,
                    help='Number of sequence samples to process before starting weight updates (default: %(default)s).')
    ap.add_argument('--up_fr', '-uf', type=int, default=2048,
                    help='For recurrent models, number of samples to run in between updating network weights, i.e the '
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
        help='Enable ground truth initialization of the first output sample in the minibatch. \n\'always\' uses teacher forcing in each minibatch;\n\'never\' never uses teacher forcing;\n\'bernoulli\' includes teacher forcing more rarely according to the fraction epochs passed.\n(default: %(default)s)')
    ap.add_argument(
        '--hidden_size',
        default=100,
        type=int,
        help='The size of the hidden layers (model-dependent) (default: %(default)s).')
    ap.add_argument('--test_sampling_rate', type=int, default=44100,
                    help='Sampling rate to use at test time. (default: %(default)s, same as in the training set).')
    ap.add_argument(
        '--save_sets',
        action='store_true',
        help='If set, the training, validation and test sets will be saved in the output folder.')
    ap.add_argument(
        '--dataset_name',
        help='Name of the dataset to use for modeling.',
        required=True)
    ap.add_argument('--nonlinearity', default='ReLU', help='Name of the torch.nn nonlinearity to use in the ODENet derivative network if that method is used (default: %(default)s).')
    ap.add_argument('--validate_every', default=1, type=int, help='Number of epochs to calculate validation loss after (default: %(default)s).')
    return ap


def train_and_test(session):
    print('Training started.')
    try:
        session.run()
    except KeyboardInterrupt:
        print('Training interrupted.')

    print('Test started.')
    try:
        test(session)
    except KeyboardInterrupt:
        print('Test interrupted, quitting.')

    close_session(session)
