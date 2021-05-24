"""Set up an LSTM-RNN architecture, run its training and test on the diode clipper data."""
import socket
from datetime import datetime
from pathlib import Path
import torch
import torchaudio
from torchdiffeq import odeint, odeint_adjoint
import CoreAudioML.networks as networks
import CoreAudioML.training as training
import CoreAudioML.dataset as dataset
from NetworkTraining import NetworkTraining
from models import StateTrajectoryNetworkFF, ODENet, ODENetDerivative


def get_run_name():
    return datetime.now().strftime(r"%B%d_%H-%M-%S") + f'_{socket.gethostname()}'

def create_dataset():
    d = dataset.DataSet(data_dir=str(Path('diode_clipper', 'data').resolve()))

    d.create_subset('train', frame_len=22050)
    d.create_subset('validation')
    d.create_subset('ignore')
    d.load_file('diodeclip', set_names=['train', 'validation', 'ignore'], splits=[0.8*0.8, 0.8*0.2, (1.0 - 0.8*0.8 - 0.8*0.2)])

    d.create_subset('test')
    d.load_file('test', set_names='test')

    return d

if __name__ == '__main__':
    session = NetworkTraining()

    session.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    session.network = networks.SimpleRNN(unit_type="LSTM", hidden_size=8, skip=0)
    # session.network = StateTrajectoryNetworkFF()
    # session.network = ODENet(ODENetDerivative(), odeint)
    session.transfer_to_device()
    session.optimizer = torch.optim.Adam(session.network.parameters(), lr=0.001)
    
    # model_directory = Path('diode_clipper', 'runs', 'odenet')
    model_directory = Path('diode_clipper', 'runs', 'lstm')
    # session.run_directory = model_directory / 'May24_13-12-09_axel'
    # session.load_checkpoint()
    run_name = get_run_name()
    session.run_directory =  model_directory / run_name

    session.loss = training.ESRLoss()
    
    session.dataset = create_dataset()

    session.epochs = 10
    session.segments_in_a_batch = 256
    session.samples_between_updates = 2048
    session.initialization_length = 1000

    session.run()

    session.device = 'cpu'
    test_output, test_loss = session.test()
    print(f'Test loss: {test_loss}')
    session.writer.add_scalar('Loss/test', test_loss, session.epochs)

    test_output_path = (session.run_directory / 'test_output.wav').resolve()
    torchaudio.save(test_output_path, test_output[None, :, 0, 0], session.dataset.subsets['test'].fs)

    if torch.cuda.is_available():
        session.writer.add_scalar('Maximum GPU memory usage [MB]', torch.cuda.max_memory_allocated('cuda') / (2 ** 20), session.epochs)

    session.writer.close()
