"""Set up an NN architecture, run its training and test on the diode clipper data."""
import argparse
from functools import partial
from pathlib import Path
import torch
import torchaudio
from torchdiffeq import odeint, odeint_adjoint
import CoreAudioML.networks as networks
import CoreAudioML.training as training
from NetworkTraining import NetworkTraining, get_run_name, create_dataset, save_json
from models import StateTrajectoryNetworkFF, ODENet, ODENetDerivative
from models.solvers import forward_euler


def argument_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', '-eps', type=int, default=20, help='Max number of training epochs to run')
    ap.add_argument('--batch_size', '-bs', type=int, default=256, help='Training mini-batch size')
    ap.add_argument('--learn_rate', '-lr', type=float, default=1e-3, help='Initial learning rate')
    ap.add_argument('--init_len', '-il', type=int, default=1000,
                  help='Number of sequence samples to process before starting weight updates')
    ap.add_argument('--up_fr', '-uf', type=int, default=2048,
                  help='For recurrent models, number of samples to run in between updating network weights, i.e the '
                       'default argument updates every 1000 samples')
    ap.add_argument('--val_chunk', '-vs', type=int, default=0, help='Number of sequence samples to process'
                                                                               'in each chunk of validation ')
    ap.add_argument('--test_chunk', '-tc', type=int, default=0, help='Number of sequence samples to process'
                                                                               'in each chunk of test')
    return ap


if __name__ == '__main__':
    args = argument_parser().parse_args()

    session = NetworkTraining()
    session.dataset = create_dataset(validation_frame_len=args.val_chunk, test_frame_len=args.test_chunk)
    sampling_rate = session.dataset.subsets['test'].fs
    
    session.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # session.network = networks.SimpleRNN(unit_type="LSTM", hidden_size=8, skip=0)
    # session.network = StateTrajectoryNetworkFF()
    # session.network = ODENet(ODENetDerivative(), partial(odeint, method='euler'), dt=1/sampling_rate)
    session.network = ODENet(ODENetDerivative(), forward_euler, dt=1/sampling_rate)
    session.transfer_to_device()
    session.optimizer = torch.optim.Adam(session.network.parameters(), lr=args.learn_rate)
    
    model_directory = Path('diode_clipper', 'runs', 'odenet')
    # model_directory = Path('diode_clipper', 'runs', 'stn')
    session.run_directory = model_directory / 'June01_19-14-15_axel'
    session.load_checkpoint()
    run_name = get_run_name()
    session.run_directory =  model_directory / run_name

    save_json(vars(args), session.run_directory / 'args.json')

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
