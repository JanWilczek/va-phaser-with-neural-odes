"""Set up an NN architecture, run its training and test on the diode clipper data."""
import argparse
import json
from functools import partial
from pathlib import Path
import torch
import torchaudio
from torchdiffeq import odeint, odeint_adjoint
import CoreAudioML.networks as networks
import CoreAudioML.training as training
from NetworkTraining import NetworkTraining, get_run_name, create_dataset, save_json
from models import StateTrajectoryNetworkFF, ODENet, ODENetDerivative
from models.solvers import forward_euler, trapezoid_rule


def argument_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('method', nargs='+')
    ap.add_argument('--epochs', '-eps', type=int, default=20, help='Max number of training epochs to run')
    ap.add_argument('--batch_size', '-bs', type=int, default=256, help='Training mini-batch size')
    ap.add_argument('--learn_rate', '-lr', type=float, default=1e-3, help='Initial learning rate')
    ap.add_argument('--cyclic_lr', '-y', action='store_true', help='If given, uses the cyclic learning rate schedule by Smith. Given learning rate parameter is used as the base learning rate, and max learning rate is 10 times larger.')
    ap.add_argument('--init_len', '-il', type=int, default=1000,
                  help='Number of sequence samples to process before starting weight updates')
    ap.add_argument('--up_fr', '-uf', type=int, default=2048,
                  help='For recurrent models, number of samples to run in between updating network weights, i.e the '
                       'default argument updates every 1000 samples')
    ap.add_argument('--val_chunk', '-vs', type=int, default=0, help='Number of sequence samples to process'
                                                                               'in each chunk of validation ')
    ap.add_argument('--test_chunk', '-tc', type=int, default=0, help='Number of sequence samples to process'
                                                                               'in each chunk of test')
    ap.add_argument('--checkpoint', '-c', type=str, default=None, help='Load a checkpoint of the given architecture with the specified name.')
    return ap


CUSTOM_SOLVERS = {'forward_euler': forward_euler,
                  'trapezoid_rule': trapezoid_rule}

def get_method(args):
    if args.method[0] == 'odenet':
        return partial(odeint, method=args.method[1])
    else:
        return CUSTOM_SOLVERS[args.method[0]]

def main():
    args = argument_parser().parse_args()

    session = NetworkTraining()
    session.dataset = create_dataset(validation_frame_len=args.val_chunk, test_frame_len=args.test_chunk)
    sampling_rate = session.dataset.subsets['test'].fs
    
    session.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # session.network = networks.SimpleRNN(unit_type="LSTM", hidden_size=8, skip=0)
    # session.network = StateTrajectoryNetworkFF()
    # session.network = ODENet(ODENetDerivative(), partial(odeint, method='euler'), dt=1/sampling_rate)
    method = get_method(args)
    session.network = ODENet(ODENetDerivative(), method, dt=1/sampling_rate)
    session.transfer_to_device()
    # session.optimizer = torch.optim.Adam(session.network.parameters(), lr=args.learn_rate)
    session.optimizer = torch.optim.SGD(session.network.parameters(), lr=args.learn_rate, momentum=0.9)
    
    model_directory = Path('diode_clipper', 'runs', 'odenet')
    # model_directory = Path('diode_clipper', 'runs', 'stn')
    
    # Untested
    if args.checkpoint is not None:
        session.run_directory = model_directory / args.checkpoint
        session.load_checkpoint()
        for param_group in session.optimizer.param_groups:
            param_group['lr'] = args.learn_rate

    if args.cyclic_lr:
        session.scheduler = torch.optim.lr_scheduler.CyclicLR(session.optimizer,
                                                                base_lr=args.learn_rate,
                                                                max_lr=10*args.learn_rate,
                                                                step_size_up=2000,
                                                                last_epoch=(session.epoch-1))

    run_name = get_run_name()
    session.run_directory =  model_directory / run_name

    save_json(vars(args), session.run_directory / 'args.json')
    session.writer.add_text('Command line arguments', json.dumps(vars(args)))

    session.loss = training.ESRLoss()
    
    session.epochs = args.epochs
    session.segments_in_a_batch = args.batch_size
    session.samples_between_updates = args.up_fr
    session.initialization_length = args.init_len

    session.run()

    session.device = 'cpu'
    test_output, test_loss = session.test()
    print(f'Test loss: {test_loss}')
    session.writer.add_scalar('Loss/test', test_loss, session.epochs)

    test_output_path = (session.run_directory / 'test_output.wav').resolve()
    torchaudio.save(test_output_path, test_output[None, :], sampling_rate)

    if torch.cuda.is_available():
        session.writer.add_scalar('Maximum GPU memory usage [MB]', torch.cuda.max_memory_allocated('cuda') / (2 ** 20), session.epochs)

    session.writer.close()


if __name__ == '__main__':
    main()
